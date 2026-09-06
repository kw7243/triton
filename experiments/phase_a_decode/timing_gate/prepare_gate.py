#!/usr/bin/env python3
"""Create immutable provenance and submission records for one timing gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contract import (
    CORRECTNESS_COMMIT,
    CORRECTNESS_PARENT,
    CORRECTNESS_TREE,
    MANIFEST_SCHEMA,
    benchmark_argv,
    snapshot,
)
from inventory import atomic_write, file_sha256


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git(repo: Path, *args: str) -> str:
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    return subprocess.check_output(
        ["git", "--no-optional-locks", "-C", str(repo), *args],
        text=True,
        env=environment,
    ).strip()


def atomic_text(path: Path, value: str, mode: int) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def capture_environment(args: argparse.Namespace) -> None:
    stage = args.stage.resolve(strict=True)
    sys.path.insert(0, str(stage / "python"))
    import torch
    import triton

    freeze = subprocess.check_output(
        [sys.executable, "-B", "-m", "pip", "freeze"],
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    atomic_text(args.lock_output, freeze, 0o600)
    executable = Path(sys.executable).resolve(strict=True)
    value = {
        "schema": "vq-phase-a-environment-identity/v1",
        "captured_at_utc": utc_now(),
        "capture_host": socket.gethostname(),
        "python_realpath": str(executable),
        "python_sha256": file_sha256(executable),
        "pip_freeze_path": str(args.lock_output.resolve()),
        "pip_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "triton": triton.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "triton_file": str(Path(triton.__file__).resolve(strict=True)),
    }
    atomic_write(args.output, value)


def stage_metadata(args: argparse.Namespace) -> None:
    source = args.source.resolve(strict=True)
    stage = args.stage.resolve(strict=True)
    commit = git(stage, "rev-parse", "HEAD")
    tree = git(stage, "rev-parse", "HEAD^{tree}")
    if commit != args.commit:
        raise RuntimeError("stage commit differs from research commit")
    if git(source, "rev-parse", "HEAD") != commit:
        raise RuntimeError("source and stage commits differ")
    value = {
        "schema": "research-reproducibility/v2",
        "created_at_utc": utc_now(),
        "hostname": socket.gethostname(),
        "source_repo": str(source),
        "staged_repo": str(stage),
        "commit": commit,
        "tree": tree,
        "branch": git(source, "branch", "--show-current"),
        "copy_mode": "git-clone-no-local-no-hardlinks",
        "command": "post-correctness Phase A timing gate; submission occurs separately",
        "includes_self_contained_git": True,
    }
    atomic_write(args.output, value)


def append_ledger(path: Path, event: str, payload: dict[str, Any]) -> dict[str, Any]:
    previous = "0" * 64
    sequence = 0
    if path.exists():
        lines = path.read_text().splitlines()
        if lines:
            prior = json.loads(lines[-1])
            previous = prior["record_sha256"]
            sequence = int(prior["sequence"]) + 1
    core = {
        "schema": "vq-phase-a-attempt-ledger/v1",
        "sequence": sequence,
        "recorded_at_utc": utc_now(),
        "event": event,
        "previous_sha256": previous,
        "payload": payload,
    }
    digest = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    record = {**core, "record_sha256": digest}
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return record


def build_manifest(args: argparse.Namespace) -> None:
    source = args.source.resolve(strict=True)
    stage = args.stage.resolve(strict=True)
    result = args.result.resolve(strict=True)
    manifest_path = args.manifest.resolve()
    request_path = args.request.resolve()
    ledger = args.ledger.resolve()
    inventory_path = args.inventory.resolve(strict=True)
    scheduler_path = args.scheduler.resolve(strict=True)
    environment_path = args.environment.resolve(strict=True)
    if any(path.parent != result for path in (manifest_path, request_path, ledger)):
        raise RuntimeError("manifest, request, and ledger must be direct result children")
    if git(source, "status", "--porcelain=v1"):
        raise RuntimeError("canonical source worktree is not clean")
    commit = git(source, "rev-parse", "HEAD")
    tree = git(source, "rev-parse", "HEAD^{tree}")
    parent = git(source, "rev-parse", "HEAD^")
    if git(source, "branch", "--show-current") != "fm/phase-a-scientific-gate":
        raise RuntimeError("wrong canonical research branch")
    if subprocess.run(
        [
            "git",
            "--no-optional-locks",
            "-C",
            str(source),
            "merge-base",
            "--is-ancestor",
            CORRECTNESS_COMMIT,
            commit,
        ],
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    ).returncode:
        raise RuntimeError("research commit is not descended from correctness checkpoint")
    metadata = json.loads((stage / "REPRODUCIBILITY_METADATA.json").read_text())
    if metadata.get("commit") != commit or metadata.get("tree") != tree:
        raise RuntimeError("stage metadata identity mismatch")
    inventory = json.loads(inventory_path.read_text())
    if inventory.get("root") != str(stage):
        raise RuntimeError("stage inventory root mismatch")
    scheduler = json.loads(scheduler_path.read_text())
    environment = json.loads(environment_path.read_text())
    selected = scheduler["selected"]
    research_root = args.research_root.resolve(strict=True)
    scratch_root = args.scratch_root.resolve(strict=True)
    if not str(result).startswith(str(research_root) + os.sep):
        raise RuntimeError("result is outside RESEARCH_ROOT")
    if not str(stage).startswith(str(scratch_root) + os.sep):
        raise RuntimeError("stage is outside SCRATCH_ROOT")
    benchmark = benchmark_argv(str(args.python.resolve(strict=True)), str(stage), str(result))
    command_path = result / "command.txt"
    atomic_text(
        command_path,
        "cd " + str(stage) + "\n" + " ".join(benchmark) + "\n",
        0o400,
    )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_at_utc": utc_now(),
        "lineage": {
            "correctness_commit": CORRECTNESS_COMMIT,
            "correctness_tree": CORRECTNESS_TREE,
            "correctness_parent": CORRECTNESS_PARENT,
            "recorded_job_1681752_classification": "PASS",
            "correctness_rerun": False,
            "research_commit": commit,
            "research_tree": tree,
            "research_parent": parent,
            "research_branch": "fm/phase-a-scientific-gate",
        },
        "paths": {
            "canonical_repo": str(source),
            "stage": str(stage),
            "result": str(result),
            "launch_manifest": str(manifest_path),
            "stage_inventory": str(inventory_path),
            "environment_identity": str(environment_path),
            "scheduler_options": str(scheduler_path),
            "attempt_ledger": str(ledger),
            "submission_request": str(request_path),
            "command": str(command_path),
        },
        "storage": {
            "research_root": str(research_root),
            "scratch_root": str(scratch_root),
            "mounts": scheduler["storage_mounts"],
            "stage_result_separated": stage != result,
        },
        "stage_inventory": {
            "path": str(inventory_path),
            "file_sha256": file_sha256(inventory_path),
            "content_digest": inventory["content_digest"],
            "record_count": inventory["record_count"],
            "content_bytes": inventory["content_bytes"],
            "git_directory": True,
            "external_worktree_pointer": False,
            "alternates": False,
            "shared_hardlinks": False,
            "frozen_read_only": True,
        },
        "benchmark": {
            "argv": benchmark,
            "command_sha256": file_sha256(command_path),
            "output_schema": "vq-phase-a-timing-result/v1",
            "contract": snapshot(),
        },
        "environment": environment,
        "scheduler": scheduler,
        "tmpdir": {
            "required_base_env": "SLURM_TMPDIR",
            "child_name": args.tmp_child,
            "must_be_allocation_local": True,
            "shared_filesystems_forbidden": [
                "nfs",
                "afs",
                "auristor",
                "cifs",
                "smb",
                "fuse.sshfs",
            ],
        },
        "execution_policy": {
            "batch_job_owner": "phase-a-scientific-gate",
            "accepted_sbatch_maximum": 1,
            "sbatch_invocations_maximum": 1,
            "gpus": 1,
            "no_requeue": True,
            "retry": False,
            "alternate_tuple": False,
            "manual_polling_after_acceptance": False,
        },
    }
    atomic_write(manifest_path, manifest, mode=0o400)
    manifest_digest = file_sha256(manifest_path)
    atomic_text(
        result / "launch_manifest.sha256",
        f"{manifest_digest}  launch_manifest.json\n",
        0o400,
    )
    stdout = str(result / "slurm-%j.out")
    sbatch = [
        "sbatch",
        "--parsable",
        "--job-name=vq-phase-a-timing-gate",
        f"--account={selected['account']}",
        f"--qos={selected['qos']}",
        f"--partition={selected['partition']}",
        f"--nodelist={selected['node']}",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={selected['cpus']}",
        "--gres=gpu:1",
        f"--mem={selected['memory']}",
        f"--time={selected['wall_time']}",
        "--no-requeue",
        f"--output={stdout}",
        str(stage / "experiments/phase_a_decode/timing_gate/run_timing.sbatch"),
    ]
    request = {
        "schema": "vq-phase-a-timing-submission/v1",
        "created_at_utc": utc_now(),
        "launch_manifest_sha256": manifest_digest,
        "launch_manifest": str(manifest_path),
        "cwd": str(stage),
        "sbatch_argv": sbatch,
        "export_policy": "default-ALL; no --export option",
        "environment": {
            "SOURCE_REPO": str(source),
            "RESEARCH_REPRO_STAGED_DIR": str(stage),
            "RESULT_DIR": str(result),
            "PHASE_A_MANIFEST_PATH": str(manifest_path),
            "PHASE_A_MANIFEST_SHA256": manifest_digest,
            "PHASE_A_PYTHON": str(args.python.resolve(strict=True)),
            "PHASE_A_CACHE_ROOT": str(args.cache_root),
            "PHASE_A_TMP_CHILD": args.tmp_child,
        },
        "benchmark_argv": benchmark,
    }
    atomic_write(request_path, request, mode=0o400)
    append_ledger(
        ledger,
        "prepared",
        {
            "launch_manifest_sha256": manifest_digest,
            "submission_request_sha256": file_sha256(request_path),
            "stage_inventory_digest": inventory["content_digest"],
            "sbatch_invocations": 0,
            "accepted_job_id": None,
        },
    )
    print(manifest_digest)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    metadata = sub.add_parser("stage-metadata")
    metadata.add_argument("--source", type=Path, required=True)
    metadata.add_argument("--stage", type=Path, required=True)
    metadata.add_argument("--commit", required=True)
    metadata.add_argument("--output", type=Path, required=True)
    capture = sub.add_parser("capture-environment")
    capture.add_argument("--stage", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--lock-output", type=Path, required=True)
    build_parser = sub.add_parser("build-manifest")
    for name in (
        "source",
        "stage",
        "result",
        "manifest",
        "request",
        "ledger",
        "inventory",
        "scheduler",
        "environment",
        "research-root",
        "scratch-root",
        "python",
        "cache-root",
    ):
        build_parser.add_argument(f"--{name}", type=Path, required=True)
    build_parser.add_argument("--tmp-child", required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command == "stage-metadata":
        stage_metadata(args)
    elif args.command == "capture-environment":
        capture_environment(args)
    else:
        build_manifest(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
