#!/usr/bin/env python3
"""Fail-closed, scheduler-free preparation for a clean Phase A rerun."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shlex
import stat
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

try:
    from .protocol import FORBIDDEN_ARTIFACTS, benchmark_argv, contract_snapshot
    from .result_protocol import atomic_write_json, atomic_write_text, read_json, utc_now
except ImportError:
    from protocol import FORBIDDEN_ARTIFACTS, benchmark_argv, contract_snapshot
    from result_protocol import atomic_write_json, atomic_write_text, read_json, utc_now


DEFAULT_APPROVED_ROOT = Path("/data/scratch-fast/kwen1/compute-native-vq")
MINIMUM_VRAM_BYTES = 8 * 1024**3
METADATA_NAME = "REPRODUCIBILITY_METADATA.json"
MANIFEST_NAME = "launch_manifest.json"
MANIFEST_DIGEST_NAME = "launch_manifest.sha256"
REQUEST_NAME = "submission_request.json"
LEDGER_NAME = "attempt_ledger.jsonl"
RUNNER = "experiments/phase_a_decode/gpu_correctness/run_gpu_correctness.sbatch"
EXECUTABLE_INPUTS = (
    "experiments/phase_a_decode/benchmark.py",
    "experiments/phase_a_decode/cpu_correctness/oracle.py",
    "experiments/phase_a_decode/cpu_correctness/test_cpu_correctness.py",
    "experiments/phase_a_decode/gpu_correctness/prepare_clean_rerun.py",
    "experiments/phase_a_decode/gpu_correctness/protocol.py",
    "experiments/phase_a_decode/gpu_correctness/result_protocol.py",
    "experiments/phase_a_decode/gpu_correctness/test_clean_rerun.py",
    RUNNER,
    "experiments/phase_a_decode/gpu_correctness/submit_from_stage.py",
    "experiments/phase_a_decode/gpu_correctness/validate_environment.py",
)
PREFLIGHT_LABELS = (
    "independent CPU oracle",
    "clean rerun behavioral guarantees",
    "fixed correctness-only protocol",
    "GPU wrapper shell syntax",
    "CPU package versions",
)


class PreparationError(RuntimeError):
    pass


def _command(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout: int = 300,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
    )


def _checked(argv: Sequence[str], *, cwd: Path, timeout: int = 300) -> str:
    completed = _command(argv, cwd=cwd, timeout=timeout)
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise PreparationError(
            f"command failed ({completed.returncode}): {shlex.join(argv)}: {detail}"
        )
    return completed.stdout.rstrip()


def _git(repo: Path, *args: str) -> str:
    return _checked(["git", "--no-optional-locks", "-C", str(repo), *args], cwd=repo)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_child(path: Path, parent: Path, label: str, *, must_exist: bool = True) -> Path:
    resolved_parent = parent.resolve(strict=False)
    resolved = path.resolve(strict=must_exist)
    if resolved == resolved_parent or resolved_parent not in resolved.parents:
        raise PreparationError(f"{label} must be a child of {resolved_parent}, got {resolved}")
    return resolved


def _git_path(repo: Path, *args: str) -> Path:
    value = Path(_git(repo, "rev-parse", *args))
    return (repo / value).resolve() if not value.is_absolute() else value.resolve()


def _status_entries(repo: Path) -> set[str]:
    completed = subprocess.run(
        [
            "git",
            "--no-optional-locks",
            "-C",
            str(repo),
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        ],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        raise PreparationError(completed.stderr.decode(errors="replace").strip())
    entries: set[str] = set()
    for raw in completed.stdout.split(b"\0"):
        if not raw:
            continue
        text = os.fsdecode(raw)
        if len(text) < 4:
            raise PreparationError(f"unparseable git status entry: {text!r}")
        entries.add(text[3:])
    return entries


def _object_records(object_dir: Path) -> dict[str, os.stat_result]:
    return {
        str(path.relative_to(object_dir)): path.stat()
        for path in object_dir.rglob("*")
        if path.is_file() and not path.is_symlink()
    }


def verify_self_contained_repo(
    repo: Path,
    expected_commit: str,
    *,
    allowed_untracked: Iterable[str] = (),
    compare_objects_with: Path | None = None,
) -> dict[str, Any]:
    repo = repo.resolve(strict=True)
    if not (repo / ".git").is_dir():
        raise PreparationError(f"{repo}/.git must be a directory, not a worktree pointer")
    top = _git_path(repo, "--show-toplevel")
    common = _git_path(repo, "--git-common-dir")
    objects = _git_path(repo, "--git-path", "objects")
    if top != repo:
        raise PreparationError(f"Git top level escapes repository: {top}")
    for label, path in (("common Git directory", common), ("object directory", objects)):
        if path != repo and repo not in path.parents:
            raise PreparationError(f"{label} escapes repository: {path}")
    alternates = objects / "info" / "alternates"
    if alternates.exists():
        raise PreparationError(f"borrowed Git object store is forbidden: {alternates}")
    head = _git(repo, "rev-parse", "HEAD")
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    if head != expected_commit:
        raise PreparationError(f"HEAD is {head}, expected {expected_commit}")
    if _git(repo, "branch", "--show-current"):
        raise PreparationError("repository must be detached at the immutable launch commit")
    status_entries = _status_entries(repo)
    expected_untracked = set(allowed_untracked)
    if status_entries != expected_untracked:
        raise PreparationError(
            f"repository status entries are {sorted(status_entries)}, "
            f"expected {sorted(expected_untracked)}"
        )
    _checked(["git", "fsck", "--full", "--no-progress"], cwd=repo)

    object_records = _object_records(objects)
    if not object_records:
        raise PreparationError("repository has no owned Git objects")
    wrong_owner = [name for name, info in object_records.items() if info.st_uid != os.getuid()]
    linked = [name for name, info in object_records.items() if info.st_nlink != 1]
    if wrong_owner:
        raise PreparationError(f"Git objects are not owned by the current user: {wrong_owner[:3]}")
    if linked:
        raise PreparationError(f"Git objects have external hard links: {linked[:3]}")

    if compare_objects_with is not None:
        other = compare_objects_with.resolve(strict=True)
        other_objects = _git_path(other, "--git-path", "objects")
        other_records = _object_records(other_objects)
        shared = []
        for name in object_records.keys() & other_records.keys():
            left, right = object_records[name], other_records[name]
            if (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino):
                shared.append(name)
        if shared:
            raise PreparationError(f"stage shares Git object inodes with source: {shared[:3]}")

    return {
        "top_level": str(top),
        "git_common_dir": str(common),
        "git_object_dir": str(objects),
        "head": head,
        "tree": tree,
        "detached": True,
        "allowed_untracked": sorted(expected_untracked),
        "object_files": len(object_records),
        "fsck": "passed",
        "objects_owned": True,
        "object_link_count_one": True,
    }


def clone_standalone_source(
    source: Path,
    destination: Path,
    expected_commit: str,
    approved_root: Path,
) -> dict[str, Any]:
    source = _canonical_child(source, approved_root, "source")
    destination = _canonical_child(
        destination, approved_root / "standalone-sources", "standalone source", must_exist=False
    )
    standalone_parent = (approved_root.resolve(strict=True) / "standalone-sources")
    if destination.parent != standalone_parent:
        raise PreparationError(f"standalone source must be a direct child of {standalone_parent}")
    if destination.exists():
        raise PreparationError(f"standalone destination already exists: {destination}")
    if _git(source, "rev-parse", "HEAD") != expected_commit:
        raise PreparationError("source is not at the requested immutable commit")
    if _status_entries(source):
        raise PreparationError("source must be clean before standalone cloning")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _checked(
        [
            "git",
            "clone",
            "--no-local",
            "--no-hardlinks",
            "--no-checkout",
            str(source),
            str(destination),
        ],
        cwd=source,
    )
    _checked(["git", "-C", str(destination), "checkout", "--detach", expected_commit], cwd=destination)
    _checked(["git", "-C", str(destination), "remote", "remove", "origin"], cwd=destination)
    identity = verify_self_contained_repo(destination, expected_commit)
    return {"source": str(source), "standalone_source": str(destination), **identity}


def _run_preflight_command(
    label: str,
    argv: list[str],
    repo: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    completed = _command(argv, cwd=repo, env=env, timeout=600)
    return {
        "label": label,
        "argv": argv,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def run_cpu_preflight(repo: Path, output: Path, expected_commit: str) -> dict[str, Any]:
    repo = repo.resolve(strict=True)
    output = output.resolve(strict=False)
    if output.exists():
        raise PreparationError(f"CPU preflight output already exists: {output}")
    report: dict[str, Any] = {
        "schema": "vq-phase-a-cpu-preflight/v2",
        "recorded_at_utc": utc_now(),
        "status": "failed",
        "repo": str(repo),
        "expected_commit": expected_commit,
        "cuda_visible_devices": "",
        "commands": [],
    }
    try:
        identity = verify_self_contained_repo(repo, expected_commit)
        report["git"] = identity
        environment = os.environ.copy()
        environment.update(
            CUDA_VISIBLE_DEVICES="",
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONHASHSEED="0",
        )
        python = sys.executable
        commands = [
            (
                "independent CPU oracle",
                [
                    python,
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    "experiments/phase_a_decode/cpu_correctness",
                    "-p",
                    "test_cpu_correctness.py",
                    "-v",
                ],
            ),
            (
                "clean rerun behavioral guarantees",
                [
                    python,
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    "experiments/phase_a_decode/gpu_correctness",
                    "-p",
                    "test_clean_rerun.py",
                    "-v",
                ],
            ),
            (
                "fixed correctness-only protocol",
                [python, "experiments/phase_a_decode/gpu_correctness/protocol.py", "--show-contract"],
            ),
            (
                "GPU wrapper shell syntax",
                ["bash", "-n", "experiments/phase_a_decode/gpu_correctness/run_gpu_correctness.sbatch"],
            ),
            (
                "CPU package versions",
                [
                    python,
                    "-c",
                    (
                        "import json,matplotlib,os,platform,torch,triton; "
                        "print(json.dumps({'python':platform.python_version(),"
                        "'torch':torch.__version__,'triton':triton.__version__,"
                        "'matplotlib':matplotlib.__version__,"
                        "'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),"
                        "'cuda_available':torch.cuda.is_available()},"
                        "sort_keys=True))"
                    ),
                ],
            ),
        ]
        report["commands"] = [
            _run_preflight_command(label, argv, repo, environment) for label, argv in commands
        ]
        report["contract"] = contract_snapshot(bf16_supported=None)
        report["input_hashes"] = {
            relative: sha256(repo / relative) for relative in EXECUTABLE_INPUTS
        }
        failed = [item["label"] for item in report["commands"] if item["returncode"] != 0]
        if failed:
            raise PreparationError("CPU preflight commands failed: " + ", ".join(failed))
        versions = json.loads(report["commands"][-1]["stdout"])
        if (
            versions.get("cuda_visible_devices") != ""
            or versions.get("cuda_available") is not False
        ):
            raise PreparationError("CPU preflight unexpectedly exposed CUDA")
        report["environment"] = versions
        report["status"] = "passed"
    except BaseException as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    atomic_write_json(output, report)
    if report["status"] != "passed":
        raise PreparationError(report.get("error", {}).get("message", "CPU preflight failed"))
    return report


def _parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise PreparationError("scheduler contract needs captured_at_utc")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError as exc:
        raise PreparationError("scheduler captured_at_utc is invalid") from exc
    if parsed.tzinfo is None:
        raise PreparationError("scheduler captured_at_utc must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def validate_scheduler_contract(contract: dict[str, Any], root: Path) -> dict[str, Any]:
    if contract.get("schema") != "vq-phase-a-scheduler-selection/v1":
        raise PreparationError("scheduler selection schema mismatch")
    captured = _parse_timestamp(contract.get("captured_at_utc"))
    age = datetime.now(timezone.utc) - captured
    if age.total_seconds() < -60 or age.total_seconds() > 15 * 60:
        raise PreparationError("scheduler selection evidence must be captured within 15 minutes")
    for key in ("account", "qos", "partition", "selection_rationale"):
        if not isinstance(contract.get(key), str) or not contract[key].strip():
            raise PreparationError(f"scheduler contract needs nonempty {key}")
    if contract.get("torralba_only") is not True:
        raise PreparationError("scheduler selection must be Torralba-only")
    if "torralba" not in contract["partition"].lower():
        raise PreparationError("selected partition must be explicitly Torralba-only")
    resources = contract.get("resources")
    expected = {"nodes": 1, "tasks": 1, "gpus": 1}
    if not isinstance(resources, dict) or any(resources.get(key) != value for key, value in expected.items()):
        raise PreparationError("scheduler resources must request one node, task, and GPU")
    if resources.get("cpus", 0) > 4 or resources.get("cpus", 0) < 1:
        raise PreparationError("scheduler request must use 1-4 CPUs")
    if resources.get("memory_gib", 0) > 16 or resources.get("memory_gib", 0) < 1:
        raise PreparationError("scheduler request must use 1-16 GiB")
    if resources.get("time_minutes", 0) > 15 or resources.get("time_minutes", 0) < 1:
        raise PreparationError("scheduler request must use 1-15 minutes")
    adequacy = contract.get("adequacy")
    if not isinstance(adequacy, dict):
        raise PreparationError("scheduler contract needs adequacy constraints")
    names = adequacy.get("advertised_gpu_names")
    if not isinstance(names, list) or not names or not all(isinstance(name, str) and name for name in names):
        raise PreparationError("adequacy needs advertised GPU names")
    if (
        not isinstance(adequacy.get("minimum_vram_bytes"), int)
        or adequacy["minimum_vram_bytes"] < MINIMUM_VRAM_BYTES
    ):
        raise PreparationError(
            f"adequacy needs at least {MINIMUM_VRAM_BYTES} bytes of predeclared VRAM"
        )
    for key in ("cuda_required", "triton_required", "exactly_one_visible_gpu"):
        if adequacy.get(key) is not True:
            raise PreparationError(f"adequacy.{key} must be true")
    if adequacy.get("bf16_policy") != "required-when-supported":
        raise PreparationError("adequacy.bf16_policy must be required-when-supported")
    evidence = contract.get("selection_evidence")
    if not isinstance(evidence, dict):
        raise PreparationError("scheduler contract needs selection evidence")
    evidence_path = root / str(evidence.get("file", ""))
    if evidence_path.is_symlink():
        raise PreparationError("scheduler evidence must not be a symlink")
    evidence_path = evidence_path.resolve(strict=True)
    if evidence_path.parent != root or not evidence_path.is_file():
        raise PreparationError("scheduler evidence must be a file in the result root")
    if evidence.get("sha256") != sha256(evidence_path):
        raise PreparationError("scheduler evidence hash mismatch")
    return contract


def _tracked_paths(repo: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(repo), "ls-files", "-z"],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        raise PreparationError(completed.stderr.decode(errors="replace").strip())
    return [os.fsdecode(item) for item in completed.stdout.split(b"\0") if item]


def _entry_digest(path: Path) -> bytes:
    if path.is_symlink():
        return hashlib.sha256(b"link\0" + os.fsencode(os.readlink(path))).digest()
    digest = hashlib.sha256(b"file\0")
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.digest()


def tracked_aggregate(repo: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    paths = _tracked_paths(repo)
    for relative in paths:
        path = repo / relative
        if not os.path.lexists(path):
            raise PreparationError(f"tracked input is missing: {relative}")
        digest.update(os.fsencode(relative))
        digest.update(b"\0")
        digest.update(_entry_digest(path))
    return len(paths), digest.hexdigest()


def _freeze_stage(stage: Path) -> None:
    paths = sorted(stage.rglob("*"), key=lambda item: len(item.parts), reverse=True)
    for path in paths:
        if path.is_symlink():
            continue
        mode = stat.S_IMODE(path.stat().st_mode)
        if path.is_dir():
            path.chmod(0o555)
        else:
            path.chmod(0o555 if mode & 0o111 else 0o444)
    stage.chmod(0o555)


def recursive_inventory(root: Path) -> list[dict[str, Any]]:
    root_info = root.lstat()
    inventory = [
        {
            "path": ".",
            "mode": format(stat.S_IMODE(root_info.st_mode), "04o"),
            "bytes": root_info.st_size,
            "type": "directory",
            "sha256": hashlib.sha256(b"directory\0").hexdigest(),
        }
    ]
    for path in sorted(root.rglob("*"), key=lambda item: os.fsencode(str(item.relative_to(root)))):
        info = path.lstat()
        entry: dict[str, Any] = {
            "path": str(path.relative_to(root)),
            "mode": format(stat.S_IMODE(info.st_mode), "04o"),
            "bytes": info.st_size,
        }
        if path.is_symlink():
            target = os.readlink(path)
            entry.update(
                type="symlink",
                target=target,
                sha256=hashlib.sha256(b"link\0" + os.fsencode(target)).hexdigest(),
            )
        elif path.is_dir():
            entry.update(
                type="directory",
                sha256=hashlib.sha256(b"directory\0").hexdigest(),
            )
        elif path.is_file():
            entry.update(type="file", sha256=sha256(path))
        else:
            raise PreparationError(f"unsupported staged path type: {path}")
        inventory.append(entry)
    return inventory


def inventory_digest(inventory: list[dict[str, Any]]) -> str:
    encoded = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _canonical_record(record: dict[str, Any]) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode()


def _ledger_record(sequence: int, previous: str | None, event: str, payload: dict[str, Any]) -> dict[str, Any]:
    record = {
        "sequence": sequence,
        "previous_sha256": previous,
        "recorded_at_utc": utc_now(),
        "event": event,
        "payload": payload,
    }
    record["record_sha256"] = hashlib.sha256(_canonical_record(record)).hexdigest()
    return record


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def create_ledger(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    record = _ledger_record(0, None, "prepared", payload)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(_canonical_record(record) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)
    return record


def read_ledger(path: Path) -> list[dict[str, Any]]:
    records = []
    previous = None
    for sequence, line in enumerate(path.read_bytes().splitlines()):
        record = json.loads(line)
        if not isinstance(record, dict):
            raise PreparationError("ledger record is not an object")
        digest = record.pop("record_sha256", None)
        actual = hashlib.sha256(_canonical_record(record)).hexdigest()
        record["record_sha256"] = digest
        if digest != actual or record.get("sequence") != sequence or record.get("previous_sha256") != previous:
            raise PreparationError("attempt ledger hash chain is invalid")
        previous = digest
        records.append(record)
    if not records:
        raise PreparationError("attempt ledger is empty")
    return records


def append_ledger(path: Path, event: str, payload: dict[str, Any]) -> dict[str, Any]:
    records = read_ledger(path)
    record = _ledger_record(len(records), records[-1]["record_sha256"], event, payload)
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
    try:
        remaining = memoryview(_canonical_record(record) + b"\n")
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise PreparationError("attempt ledger append made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return record


def _sbatch_argv(
    scheduler: dict[str, Any],
    result: Path,
) -> list[str]:
    resources = scheduler["resources"]
    return [
        "sbatch",
        "--parsable",
        "--no-requeue",
        f"--account={scheduler['account']}",
        f"--qos={scheduler['qos']}",
        f"--partition={scheduler['partition']}",
        f"--nodes={resources['nodes']}",
        f"--ntasks={resources['tasks']}",
        f"--cpus-per-task={resources['cpus']}",
        f"--gres=gpu:{resources['gpus']}",
        f"--mem={resources['memory_gib']}G",
        f"--time=00:{resources['time_minutes']:02d}:00",
        f"--output={result}/slurm-%j.out",
        f"--error={result}/slurm-%j.out",
        RUNNER,
    ]


def _sbatch_environment(
    source: Path,
    stage: Path,
    result: Path,
    manifest_sha256: str,
    python: str,
) -> dict[str, str]:
    return {
        "SOURCE_REPO": str(source),
        "RESEARCH_REPRO_STAGED_DIR": str(stage),
        "RESULT_DIR": str(result),
        "PHASE_A_MANIFEST_PATH": str(result / MANIFEST_NAME),
        "PHASE_A_MANIFEST_SHA256": manifest_sha256,
        "PHASE_A_PYTHON": python,
    }


def prepare_frozen_stage(
    source: Path,
    stage: Path,
    result: Path,
    cpu_preflight: Path,
    scheduler_contract_path: Path,
    expected_commit: str,
    expected_tree: str,
    python: str,
    approved_root: Path,
) -> dict[str, Any]:
    source = _canonical_child(source, approved_root / "standalone-sources", "standalone source")
    stage = _canonical_child(stage, approved_root / "staging", "stage")
    result = _canonical_child(result, approved_root / "results", "result")
    resolved_root = approved_root.resolve(strict=True)
    expected_parents = {
        "standalone source": resolved_root / "standalone-sources",
        "stage": resolved_root / "staging",
        "result": resolved_root / "results",
    }
    for label, path in (("standalone source", source), ("stage", stage), ("result", result)):
        if path.parent != expected_parents[label]:
            raise PreparationError(f"{label} must be a direct child of {expected_parents[label]}")
        if path.stat().st_uid != os.getuid():
            raise PreparationError(f"{label} is not owned by the current user")
    if stat.S_IMODE(result.stat().st_mode) != 0o700:
        raise PreparationError("fresh result root must have mode 0700")
    cpu_preflight = cpu_preflight.resolve(strict=True)
    scheduler_contract_path = scheduler_contract_path.resolve(strict=True)
    if len({source, stage, result}) != 3:
        raise PreparationError("source, stage, and result paths must be distinct")
    if cpu_preflight.parent != result or scheduler_contract_path.parent != result:
        raise PreparationError("preflight and scheduler contract must be in the result root")

    source_identity = verify_self_contained_repo(source, expected_commit)
    stage_identity = verify_self_contained_repo(
        stage,
        expected_commit,
        allowed_untracked=(METADATA_NAME,),
        compare_objects_with=source,
    )
    if source_identity["tree"] != expected_tree or stage_identity["tree"] != expected_tree:
        raise PreparationError("source/stage tree does not match the declared launch tree")

    metadata_path = stage / METADATA_NAME
    metadata = read_json(metadata_path)
    required_metadata = {
        "source_repo": str(source),
        "staged_repo": str(stage),
        "git_commit_full": expected_commit,
        "git_status_short": "",
        "cwd": str(source),
    }
    for key, value in required_metadata.items():
        if metadata.get(key) != value:
            raise PreparationError(f"reproducibility metadata {key} mismatch")
    short_commit = metadata.get("git_commit_short")
    if (
        not isinstance(short_commit, str)
        or not re.fullmatch(r"[0-9a-f]{7,40}", short_commit)
        or not expected_commit.startswith(short_commit)
    ):
        raise PreparationError("reproducibility metadata short commit mismatch")
    created_at = _parse_timestamp(metadata.get("created_at_utc"))
    if created_at > datetime.now(timezone.utc) + timedelta(minutes=1):
        raise PreparationError("reproducibility metadata creation time is in the future")
    for key in ("command", "hostname"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise PreparationError(f"reproducibility metadata needs nonempty {key}")
    try:
        metadata_command = shlex.split(metadata["command"])
    except ValueError as exc:
        raise PreparationError("reproducibility metadata command is malformed") from exc
    if "freeze-stage" not in metadata_command:
        raise PreparationError("reproducibility metadata command is not the stage verifier")

    preflight = read_json(cpu_preflight)
    if (
        preflight.get("schema") != "vq-phase-a-cpu-preflight/v2"
        or preflight.get("status") != "passed"
        or preflight.get("expected_commit") != expected_commit
        or preflight.get("repo") != str(source)
    ):
        raise PreparationError("CPU preflight did not pass for the launch commit")
    if (
        preflight.get("git", {}).get("head") != expected_commit
        or preflight.get("git", {}).get("tree") != expected_tree
    ):
        raise PreparationError("CPU preflight tree mismatch")
    environment_identity = preflight.get("environment")
    if (
        not isinstance(environment_identity, dict)
        or environment_identity.get("cuda_visible_devices") != ""
        or environment_identity.get("cuda_available") is not False
    ):
        raise PreparationError("CPU preflight environment identity is missing")
    commands = preflight.get("commands")
    if (
        not isinstance(commands, list)
        or [command.get("label") for command in commands if isinstance(command, dict)]
        != list(PREFLIGHT_LABELS)
        or any(command.get("returncode") != 0 for command in commands)
    ):
        raise PreparationError("CPU preflight command evidence is missing")
    preflight_argv = commands[0].get("argv")
    if not isinstance(preflight_argv, list) or not preflight_argv:
        raise PreparationError("CPU preflight Python identity is missing")
    python_path = Path(python).resolve(strict=True)
    if python_path != Path(preflight_argv[0]).resolve(strict=True) or not os.access(python_path, os.X_OK):
        raise PreparationError("launch Python differs from the CPU preflight interpreter")
    expected_input_hashes = {
        relative: sha256(source / relative) for relative in EXECUTABLE_INPUTS
    }
    if preflight.get("input_hashes") != expected_input_hashes:
        raise PreparationError("CPU preflight executable hashes differ from the source")
    if preflight.get("contract") != contract_snapshot(bf16_supported=None):
        raise PreparationError("CPU preflight correctness contract differs from the launch contract")
    scheduler = validate_scheduler_contract(read_json(scheduler_contract_path), result)

    initial_result_files = {path.name for path in result.iterdir()}
    scheduler_evidence = scheduler["selection_evidence"]["file"]
    expected_initial = {cpu_preflight.name, scheduler_contract_path.name, scheduler_evidence}
    if initial_result_files != expected_initial:
        raise PreparationError(
            f"result root is not fresh: found {sorted(initial_result_files)}, "
            f"expected {sorted(expected_initial)}"
        )

    source_count, source_hash = tracked_aggregate(source)
    stage_count, stage_hash = tracked_aggregate(stage)
    if (source_count, source_hash) != (stage_count, stage_hash):
        raise PreparationError("source/stage tracked content differs")

    _freeze_stage(stage)
    stage_identity = verify_self_contained_repo(
        stage,
        expected_commit,
        allowed_untracked=(METADATA_NAME,),
        compare_objects_with=source,
    )
    inventory = recursive_inventory(stage)
    executables = {relative: sha256(stage / relative) for relative in EXECUTABLE_INPUTS}
    manifest = {
        "schema": "vq-phase-a-clean-rerun-launch/v2",
        "prepared_at_utc": utc_now(),
        "commit": expected_commit,
        "tree": expected_tree,
        "source_repo": str(source),
        "staged_repo": str(stage),
        "result_root": str(result),
        "stage_read_only": True,
        "source_identity": source_identity,
        "stage_identity": stage_identity,
        "tracked_files": stage_count,
        "tracked_content_sha256": stage_hash,
        "stage_inventory": inventory,
        "stage_inventory_sha256": inventory_digest(inventory),
        "reproducibility_metadata_sha256": sha256(metadata_path),
        "cpu_preflight": {"file": cpu_preflight.name, "sha256": sha256(cpu_preflight)},
        "environment_identity": environment_identity,
        "scheduler_selection": scheduler,
        "scheduler_contract_sha256": sha256(scheduler_contract_path),
        "executable_hashes": executables,
        "benchmark_argv": benchmark_argv(str(python_path), str(stage), str(result)),
        "correctness_contract": contract_snapshot(bf16_supported=None),
        "expected_result_schema": {
            "schema": "vq-phase-a-gpu-correctness-r1/v2",
            "required_runtime_artifacts": [
                "runtime_start.json",
                "environment_validation.json",
                "environment.lock.txt",
                "command.txt",
                "run_metadata.json",
                "correctness.json",
                "correctness_manifest.json",
                "README.md",
                "execution_state.json",
                "final_result_manifest.json",
            ],
            "classifications": ["PASS", "FAIL", "NO RESULT"],
            "failure_phase_required_for_non_pass": True,
        },
        "forbidden_artifacts": list(FORBIDDEN_ARTIFACTS),
        "attempt_policy": {
            "maximum_sbatch_invocations": 1,
            "retry": False,
            "requeue": False,
            "cancellation_after_acceptance": False,
        },
    }
    manifest_path = result / MANIFEST_NAME
    atomic_write_json(manifest_path, manifest, mode=0o400)
    manifest_path.chmod(0o444)
    manifest_sha256 = sha256(manifest_path)
    digest_path = result / MANIFEST_DIGEST_NAME
    atomic_write_text(digest_path, f"{manifest_sha256}  {MANIFEST_NAME}\n", mode=0o400)
    digest_path.chmod(0o444)

    request = {
        "schema": "vq-phase-a-submission-request/v2",
        "launch_manifest_sha256": manifest_sha256,
        "sbatch_argv": _sbatch_argv(scheduler, result),
        "sbatch_environment": _sbatch_environment(
            source, stage, result, manifest_sha256, str(python_path)
        ),
        "benchmark_argv": manifest["benchmark_argv"],
    }
    request_path = result / REQUEST_NAME
    atomic_write_json(request_path, request, mode=0o400)
    request_path.chmod(0o444)
    create_ledger(
        result / LEDGER_NAME,
        {
            "launch_manifest_sha256": manifest_sha256,
            "submission_request_sha256": sha256(request_path),
            "accepted_job_id": None,
            "sbatch_invocations": 0,
        },
    )
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "request": str(request_path),
        "ledger": str(result / LEDGER_NAME),
        "stage": str(stage),
    }


def submit_once(
    manifest_path: Path,
    request_path: Path,
    ledger_path: Path,
    *,
    runner: Callable[
        [list[str], Path, dict[str, str]], subprocess.CompletedProcess[Any]
    ] | None = None,
) -> str:
    manifest_path = manifest_path.resolve(strict=True)
    request_path = request_path.resolve(strict=True)
    ledger_path = ledger_path.resolve(strict=True)
    root = manifest_path.parent
    if manifest_path.name != MANIFEST_NAME or request_path != root / REQUEST_NAME:
        raise PreparationError("manifest and request must use their prepared result-root paths")
    if ledger_path != root / LEDGER_NAME:
        raise PreparationError("ledger must use its prepared result-root path")
    manifest = read_json(manifest_path)
    request = read_json(request_path)
    if root != Path(manifest.get("result_root", "")).resolve(strict=True):
        raise PreparationError("prepared result root does not match the launch manifest")
    manifest_sha256 = sha256(manifest_path)
    if request.get("launch_manifest_sha256") != manifest_sha256:
        raise PreparationError("submission request is not bound to the launch manifest")
    if Path.cwd().resolve() != Path(manifest["staged_repo"]).resolve(strict=True):
        raise PreparationError("submission must run from the frozen staged repository")
    current_inventory = recursive_inventory(Path(manifest["staged_repo"]))
    if inventory_digest(current_inventory) != manifest.get("stage_inventory_sha256"):
        raise PreparationError("frozen stage inventory changed after preparation")
    expected_request = _sbatch_argv(
        manifest["scheduler_selection"],
        Path(manifest["result_root"]),
    )
    expected_environment = _sbatch_environment(
        Path(manifest["source_repo"]),
        Path(manifest["staged_repo"]),
        Path(manifest["result_root"]),
        manifest_sha256,
        manifest["benchmark_argv"][0],
    )
    if request.get("sbatch_argv") != expected_request:
        raise PreparationError("submission argv differs from the immutable manifest")
    if request.get("sbatch_environment") != expected_environment:
        raise PreparationError("submission environment differs from the immutable manifest")
    if any(argument.startswith("--export") for argument in expected_request):
        raise PreparationError("submission argv must use sbatch's safe default ALL export")
    if request.get("benchmark_argv") != manifest.get("benchmark_argv"):
        raise PreparationError("benchmark argv differs from the immutable manifest")
    validate_scheduler_contract(manifest["scheduler_selection"], root)
    records = read_ledger(ledger_path)
    prepared = records[0].get("payload", {})
    if (
        prepared.get("launch_manifest_sha256") != manifest_sha256
        or prepared.get("submission_request_sha256") != sha256(request_path)
        or prepared.get("sbatch_invocations") != 0
        or prepared.get("accepted_job_id") is not None
    ):
        raise PreparationError("attempt ledger is not bound to the prepared request")
    if any(record["event"] in {"submission_started", "submission_finished", "accepted"} for record in records):
        raise PreparationError("attempt ledger already records a submission; retry is forbidden")

    lock = root / ".launch.lock"
    lock_fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    os.fsync(lock_fd)
    os.close(lock_fd)
    _fsync_directory(root)
    argv = request.get("sbatch_argv")
    if not isinstance(argv, list) or not argv or argv[0] != "sbatch" or not all(
        isinstance(item, str) for item in argv
    ):
        raise PreparationError("submission request has invalid sbatch argv")
    append_ledger(
        ledger_path,
        "submission_started",
        {
            "sbatch_argv": argv,
            "sbatch_environment": expected_environment,
            "accepted_job_id_before": None,
        },
    )
    if runner is None:
        def runner(
            command: list[str], cwd: Path, environment: dict[str, str]
        ) -> subprocess.CompletedProcess[bytes]:
            return subprocess.run(
                command,
                cwd=cwd,
                env={**os.environ, **environment},
                capture_output=True,
                check=False,
            )
    completed = runner(argv, Path(manifest["staged_repo"]), expected_environment)
    stdout_bytes = (
        completed.stdout.encode("utf-8")
        if isinstance(completed.stdout, str)
        else completed.stdout
    )
    stderr_bytes = (
        completed.stderr.encode("utf-8")
        if isinstance(completed.stderr, str)
        else completed.stderr
    )
    response = {
        "returncode": completed.returncode,
        "stdout_base64": base64.b64encode(stdout_bytes).decode("ascii"),
        "stdout_sha256": hashlib.sha256(stdout_bytes).hexdigest(),
        "stderr_base64": base64.b64encode(stderr_bytes).decode("ascii"),
        "stderr_sha256": hashlib.sha256(stderr_bytes).hexdigest(),
    }
    append_ledger(ledger_path, "submission_finished", response)
    try:
        job_id = stdout_bytes.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise PreparationError("sbatch returned non-ASCII bytes; retry is forbidden") from exc
    if completed.returncode != 0:
        raise PreparationError(f"sbatch failed with {completed.returncode}; retry is forbidden")
    if re.fullmatch(r"[0-9]+", job_id) is None:
        raise PreparationError("sbatch returned a non-numeric response; retry is forbidden")
    append_ledger(ledger_path, "accepted", {"job_id": job_id})
    return job_id


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    clone = subparsers.add_parser("clone-source")
    clone.add_argument("--source", type=Path, required=True)
    clone.add_argument("--destination", type=Path, required=True)
    clone.add_argument("--commit", required=True)
    clone.add_argument("--approved-root", type=Path, default=DEFAULT_APPROVED_ROOT)

    preflight = subparsers.add_parser("cpu-preflight")
    preflight.add_argument("--repo", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)
    preflight.add_argument("--commit", required=True)

    freeze = subparsers.add_parser("freeze-stage")
    freeze.add_argument("--source", type=Path, required=True)
    freeze.add_argument(
        "--stage",
        type=Path,
        default=Path(os.environ["RESEARCH_REPRO_STAGED_DIR"])
        if os.environ.get("RESEARCH_REPRO_STAGED_DIR") else None,
    )
    freeze.add_argument("--result", type=Path, required=True)
    freeze.add_argument("--cpu-preflight", type=Path, required=True)
    freeze.add_argument("--scheduler-contract", type=Path, required=True)
    freeze.add_argument("--commit", required=True)
    freeze.add_argument("--tree", required=True)
    freeze.add_argument("--python", required=True)
    freeze.add_argument("--approved-root", type=Path, default=DEFAULT_APPROVED_ROOT)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "clone-source":
        result = clone_standalone_source(args.source, args.destination, args.commit, args.approved_root)
    elif args.command == "cpu-preflight":
        result = run_cpu_preflight(args.repo, args.output, args.commit)
    else:
        if args.stage is None:
            raise PreparationError(
                "--stage or RESEARCH_REPRO_STAGED_DIR is required for freeze-stage"
            )
        result = prepare_frozen_stage(
            args.source,
            args.stage,
            args.result,
            args.cpu_preflight,
            args.scheduler_contract,
            args.commit,
            args.tree,
            args.python,
            args.approved_root,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
