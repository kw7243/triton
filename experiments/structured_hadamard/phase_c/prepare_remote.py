#!/usr/bin/env python3
"""Prepare the single remote Phase C attempt and exact staged clearance."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Mapping

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.structured_hadamard.phase_a.real_model import preflight_remote_audit as base
from experiments.structured_hadamard.phase_c import preflight, selector


LEDGER_ZERO = {
    "owner_attempts": 0,
    "salloc_attempts": 0,
    "srun_attempts": 0,
    "driver_attempts": 0,
    "terminal_events_fired": 0,
}


class PreparationError(RuntimeError):
    """The immutable stage or attempt inputs cannot be bound exactly."""


def _exclusive_json(path: Path, value: object, *, mode: int = 0o600) -> None:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _git(root: Path, *argv: str) -> str:
    completed = subprocess.run(("git", *argv), cwd=root, check=False, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode:
        raise PreparationError(f"git {' '.join(argv)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _make_stage_immutable(stage: Path) -> None:
    manifest = json.loads((stage / "REPRODUCIBILITY_MANIFEST.json").read_text(encoding="utf-8"))
    for relative, identity in manifest["entries"].items():
        path = stage / relative
        if identity["kind"] == "file":
            mode = stat.S_IMODE(path.stat().st_mode)
            path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    for name in ("REPRODUCIBILITY_MANIFEST.json", "REPRODUCIBILITY_METADATA.json"):
        path = stage / name
        path.chmod(stat.S_IMODE(path.stat().st_mode) & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _stage_record(stage: Path) -> dict[str, object]:
    stage = stage.resolve(strict=True)
    metadata_path = stage / "REPRODUCIBILITY_METADATA.json"
    manifest_path = stage / "REPRODUCIBILITY_MANIFEST.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    head = _git(stage, "rev-parse", "HEAD^{commit}")
    tree = _git(stage, "rev-parse", "HEAD^{tree}")
    if metadata["source_head"] != head:
        raise PreparationError("stage metadata source head differs")
    if base.canonical_sha256(manifest["entries"]) != metadata["working_tree_manifest_sha256"]:
        raise PreparationError("stage manifest canonical digest differs")
    return {
        "root": str(stage), "head": head, "tree": tree,
        "required_ancestor": selector.PHASE_B_PROVENANCE_TIP,
        "metadata_path": str(metadata_path),
        "metadata_sha256": base.sha256_file(metadata_path),
        "manifest": {
            "manifest_path": str(manifest_path),
            "manifest_file_sha256": base.sha256_file(manifest_path),
            "entries_sha256": metadata["working_tree_manifest_sha256"],
            "immutable": True,
        },
    }


def _file_record(path: Path) -> dict[str, object]:
    path = path.resolve(strict=True)
    return {
        "path": str(path), "bytes": path.stat().st_size,
        "mode": stat.S_IMODE(path.stat().st_mode), "sha256": base.sha256_file(path),
    }


def _helper(name: str, path: Path) -> dict[str, object]:
    return {"name": name, **_file_record(path)}


def _data_file(data: Mapping[str, object], name: str) -> str:
    matches = [row for row in data["files"] if row["name"] == name]
    if len(matches) != 1:
        raise PreparationError(f"accepted data record lacks exactly one {name}")
    return str(matches[0]["resolved_path"])


def prepare(args: argparse.Namespace) -> dict[str, object]:
    attempt = args.attempt_root.resolve(strict=True)
    stage = args.project_stage.resolve(strict=True)
    if attempt not in stage.parents:
        raise PreparationError("project stage must be inside the exact attempt root")
    stage_status = set(_git(stage, "status", "--porcelain=v1", "--untracked-files=all").splitlines())
    if stage_status != {
            "?? REPRODUCIBILITY_MANIFEST.json", "?? REPRODUCIBILITY_METADATA.json"}:
        raise PreparationError("fresh stage has unexpected working-tree state")
    phase_b_clearance = json.loads(args.phase_b_clearance.read_text(encoding="utf-8"))
    phase_b_preflight = json.loads(args.phase_b_preflight.read_text(encoding="utf-8"))
    if phase_b_preflight.get("status") != "passed":
        raise PreparationError("accepted Phase B preflight result is not passed")

    run_state = attempt / "run-state"
    provenance = attempt / "provenance"
    output_parent = attempt / "output-parent"
    runtime_cache = attempt / "runtime-cache"
    for path, mode in ((run_state, 0o700), (provenance, 0o700), (output_parent, 0o700),
                       (runtime_cache, 0o700)):
        path.mkdir(mode=mode, exist_ok=False)
    for name in ("huggingface", "torch", "xdg"):
        (runtime_cache / name).mkdir(mode=0o700)

    _make_stage_immutable(stage)
    project = _stage_record(stage)
    policy_path = stage / "data/rot-phasec-selector-r1/policy-freeze.json"
    map_path = stage / selector.PHASE_B_MAP_REPO_PATH
    report_path = stage / selector.PHASE_B_REPORT_REPO_PATH
    results_path = stage / "data/rot-phaseb-map-r1/artifacts/run-1662528/scientific-run/results.json"
    freeze = json.loads(policy_path.read_text(encoding="utf-8"))
    selector.validate_freeze(freeze, map_path=map_path, report_path=report_path)
    policy_sha256 = base.sha256_file(policy_path)

    environment = copy.deepcopy(phase_b_clearance["environment"])
    observed_packages = {row["name"].lower(): row for row in phase_b_preflight["environment"]["packages"]}
    for package in environment["packages"]:
        observed = observed_packages.get(str(package["name"]).lower())
        if not observed or "byte_manifest_sha256" not in observed:
            raise PreparationError(f"Phase B preflight lacks exact package bytes for {package['name']}")
        package["byte_manifest_sha256"] = observed["byte_manifest_sha256"]
    environment["runtime_env"].update({
        "HF_HOME": str(runtime_cache / "huggingface"),
        "TORCH_HOME": str(runtime_cache / "torch"),
        "XDG_CACHE_HOME": str(runtime_cache / "xdg"),
    })
    python = environment["python"]["path"]
    status_path = run_state / "task-status.jsonl"
    status_path.touch(mode=0o600, exist_ok=False)
    ledger_path = run_state / "owner-ledger.json"
    initial_ledger_path = run_state / "owner-ledger.initial.json"
    _exclusive_json(initial_ledger_path, LEDGER_ZERO)
    _exclusive_json(ledger_path, LEDGER_ZERO)
    output_directory = output_parent / "scientific-run"
    clearance_path = run_state / "clearance.json"
    terminal_path = run_state / "remote-terminal.json"
    audit_output = run_state / "preflight-audit.json"

    driver = stage / "experiments/structured_hadamard/phase_c/driver.py"
    owner = stage / "experiments/structured_hadamard/phase_c/gpu_owner.py"
    driver_argv = [
        python, str(driver),
        "--stage-root", str(stage),
        "--source-commit", project["head"],
        "--snapshot", phase_b_clearance["model"]["root"],
        "--model-repository", phase_a_model_repository(),
        "--model-revision", phase_b_clearance["model"]["revision"],
        "--evaluation-arrow", _data_file(phase_b_clearance["data"], "wikitext-test.arrow"),
        "--evaluation-sha256", driver_evaluation_sha256(),
        "--dataset-revision", phase_b_clearance["data"]["revision"],
        "--extension", phase_b_clearance["extension"]["path"],
        "--extension-sha256", selector_extension_sha256(),
        "--quarot-root", phase_b_clearance["dependency"]["root"],
        "--phase-a-results", str(stage / "data/rot-phasea-real-model-baseline-r1/artifacts/run-1662352/results.json"),
        "--phase-b-map", str(map_path),
        "--phase-b-report", str(report_path),
        "--policy-freeze", str(policy_path),
        "--policy-freeze-sha256", policy_sha256,
        "--output-directory", str(output_directory),
        "--status-path", str(status_path),
        "--device", "cuda:0", "--seed", "20260902",
        "--ppl-sequence-length", "1024", "--prompt-length", "128",
        "--output-length", "32", "--decode-warmups", "1",
        "--decode-repetitions", "5", "--fusion", "none",
    ]
    owner_argv = [python, str(owner), "--owner", str(clearance_path)]
    allocated_argv = [python, str(owner), "--allocated", str(clearance_path)]
    salloc_argv = [
        "/usr/bin/salloc", "--account=vision-torralba-urops-meng",
        "--qos=vision-torralba-interactive", f"--partition={args.partition}",
        "--job-name=rot-phasec-selector-r1", "--nodes=1", "--ntasks=1",
        "--cpus-per-task=2", "--mem=32G", "--gres=gpu:1", "--time=00:45:00",
    ]
    srun_argv = [
        "/usr/bin/srun", "--pty", "--nodes=1", "--ntasks=1", "--cpus-per-task=2",
        "--mem=32G", "--gres=gpu:1", "--kill-on-bad-exit=1",
    ]
    argv_record = {
        "owner_argv": owner_argv, "allocated_argv": allocated_argv,
        "driver_argv": driver_argv, "salloc_argv": salloc_argv, "srun_argv": srun_argv,
    }
    argv_path = provenance / "execution-argv.json"
    _exclusive_json(argv_path, argv_record)
    argv_path.chmod(0o400)

    helper_paths = {
        "phase_c_preflight": stage / "experiments/structured_hadamard/phase_c/preflight.py",
        "phase_c_driver": driver,
        "gpu_owner": owner,
        "local_event_owner": stage / "experiments/structured_hadamard/phase_c/local_event_owner.py",
        "phase_c_selector": stage / "experiments/structured_hadamard/phase_c/selector.py",
        "phase_c_analysis": stage / "experiments/structured_hadamard/phase_c/analysis.py",
        "phase_c_schema": stage / "experiments/structured_hadamard/phase_c/schema.py",
        "phase_c_verifier": stage / "experiments/structured_hadamard/phase_c/verify_results.py",
        "phase_a_benchmark": stage / "experiments/structured_hadamard/phase_a/real_model/benchmark.py",
        "phase_a_runtime": stage / "experiments/structured_hadamard/phase_a/real_model/runtime.py",
        "phase_b_runtime": stage / "experiments/structured_hadamard/phase_b/runtime.py",
        "phase_b_reference": stage / "experiments/structured_hadamard/phase_b/reference.py",
        "execution_argv": argv_path,
    }
    helpers = [_helper(name, path) for name, path in helper_paths.items()]
    execution = {
        **argv_record,
        "argv_sha256": base.canonical_sha256(argv_record),
        "duration_estimate": {
            "Phase_A_observed": "00:05:34 for three PPL/decode variants",
            "Phase_B_observed": "00:14:39 for map, six PPL variants, and cache",
            "estimated_runtime": "00:20:00 for seven deduplicated PPL/decode policies",
            "buffer": "00:25:00", "requested": "00:45:00",
            "host_memory_reason": "32G is the smallest previously demonstrated real-model request; Phase C omits the 9.1 GiB Phase B cache and releases each policy model sequentially",
        },
        "output_parent": str(output_parent),
        "output_parent_mode": 0o700,
        "output_directory": str(output_directory),
    }
    plan = _file_record(args.authoritative_plan)
    phase_b = {
        "map": _file_record(map_path),
        "report": _file_record(report_path),
        "results": _file_record(results_path),
    }
    clearance = {
        "schema_version": preflight.SCHEMA_VERSION,
        "owner": {"name": os.environ.get("USER"), "uid": os.getuid()},
        "project": project,
        "accepted_project": phase_b_clearance["project"],
        "dependency": phase_b_clearance["dependency"],
        "extension": phase_b_clearance["extension"],
        "model": phase_b_clearance["model"],
        "data": phase_b_clearance["data"],
        "environment": environment,
        "helpers": helpers,
        "execution": execution,
        "ledger": {
            "path": str(ledger_path), "expected_zero": LEDGER_ZERO,
            "status_path": str(status_path), "status_sha256": base.sha256_file(status_path),
        },
        "terminal_path": str(terminal_path),
        "audit_output": str(audit_output),
        "plan": plan,
        "phase_b": phase_b,
        "policy": {"freeze": _file_record(policy_path), "freeze_sha256": policy_sha256},
    }
    _exclusive_json(clearance_path, clearance)
    return {
        "schema_version": "phase-c-attempt-preparation-v1",
        "attempt_root": str(attempt),
        "project_stage": project,
        "policy_freeze_sha256": policy_sha256,
        "clearance": str(clearance_path),
        "clearance_sha256": base.sha256_file(clearance_path),
        "preflight_helper_sha256": base.sha256_file(helper_paths["phase_c_preflight"]),
        "preflight_output": str(audit_output),
        "execution_argv": str(argv_path),
        "execution_argv_sha256": base.sha256_file(argv_path),
        "ledger_initial": str(initial_ledger_path),
        "resources": {"partition": args.partition, "cpus": 2, "memory": "32G", "time": "00:45:00"},
    }


def phase_a_model_repository() -> str:
    return "NousResearch/Meta-Llama-3-8B"


def driver_evaluation_sha256() -> str:
    return "2b8a3efac7b468cbe6432edba5f55c21e435d93873acc6727431f08d5ed328ea"


def selector_extension_sha256() -> str:
    return "10a961e8855d7349708412aa8c21a38667fb75c94c372180dd6f992913e0a8e0"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--project-stage", type=Path, required=True)
    parser.add_argument("--phase-b-clearance", type=Path, required=True)
    parser.add_argument("--phase-b-preflight", type=Path, required=True)
    parser.add_argument("--authoritative-plan", type=Path, required=True)
    parser.add_argument("--partition", required=True)
    args = parser.parse_args(argv)
    try:
        result = prepare(args)
    except (PreparationError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
