#!/usr/bin/env python3
"""CPU-only, scheduler-free validation of the frozen Phase A launch contract."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "contract.json"
BRANCH = "fm/phase-a-benchmark-preflight"
BASE = "f33a9c88651a1defe2b0a7ef4f80a3cfa1a8f25b"
SOURCE = "/data/scratch-fast/kwen1/compute-native-vq/triton"
WORKTREE = "/data/scratch-fast/kwen1/compute-native-vq/worktrees/vq-benchmark-preflight-r1"
STAGING_PARENT = "/data/scratch-fast/kwen1/compute-native-vq/staging"
RESULT_DIR = SOURCE + "/results/2026-08-20-hurwitz-decode-baseline"
FROZEN_FILES = {
    "experiments/phase_a_decode/benchmark.py":
        "c2c451ab3290af8d4fc97ff8bec077533a679ffd825a2d44d8907dc8cdd16a6a",
    "experiments/phase_a_decode/run_phase_a.sbatch":
        "aca356cc4d1ba87ea67c94f6bca671e52e0a26b74a779722fae218dd01ba8bc6",
}
DIRECTIVES = {
    "account": "vision-torralba-urops-meng",
    "qos": "vision-torralba-interactive",
    "partition": "vision-torralba-rtx3090",
    "nodes": "1", "ntasks": "1", "cpus-per-task": "4", "gres": "gpu:1",
    "mem": "16G", "time": "01:00:00",
    "output": RESULT_DIR + "/slurm-%j.out",
    "error": RESULT_DIR + "/slurm-%j.out",
}
ARTIFACTS = [
    "slurm-<jobid>.out", "staged_snapshot.txt", "REPRODUCIBILITY_METADATA.json",
    "environment.lock.txt", "system.txt", "command.txt", "run_metadata.json",
    "correctness.json", "tuning.json", "trial_timings.json", "timings.csv",
    "README.md", "jh_speedup.png",
]
METADATA_FIELDS = [
    "command", "created_at_utc", "cwd", "git_commit_full", "git_commit_short",
    "git_status_short", "hostname", "source_repo", "staged_repo",
]


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], cwd: Path) -> str:
    completed = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, check=False)
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(
            f"command failed ({completed.returncode}): {' '.join(command)}: {detail}")
    return completed.stdout.rstrip()


def git(repo: Path, *args: str) -> str:
    return run(["git", *args], repo)


def parse_directives(text: str) -> tuple[dict[str, str], list[str]]:
    values: dict[str, str] = {}
    errors: list[str] = []
    for line in text.splitlines():
        if not line.startswith("#SBATCH --"):
            continue
        item = line[len("#SBATCH --"):]
        if "=" not in item:
            errors.append(f"Slurm directive must use --key=value: {line}")
            continue
        key, value = item.split("=", 1)
        if key in values:
            errors.append(f"duplicate Slurm directive: {key}")
        values[key] = value
    return values, errors


def get_dotted(value: dict[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        current = current.get(part) if isinstance(current, dict) else None
    return current


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected = {
        "schema_version": 1, "provenance.branch": BRANCH,
        "provenance.base_commit": BASE, "provenance.source_repo": SOURCE,
        "provenance.worktree": WORKTREE, "provenance.frozen_files": FROZEN_FILES,
        "staging.mode": "full-repository-copy",
        "staging.staging_parent": STAGING_PARENT,
        "staging.required_metadata_fields": METADATA_FIELDS,
        "slurm.directives": DIRECTIVES, "results.directory": RESULT_DIR,
        "results.artifacts": ARTIFACTS,
    }
    for dotted, wanted in expected.items():
        current = get_dotted(manifest, dotted)
        if current != wanted:
            errors.append(f"manifest {dotted} is {current!r}, expected {wanted!r}")
    lane = manifest.get("lane", {})
    for key in ("scheduler_contact", "submission_authorized",
                "fourth_phase_a_submission_authorized", "uid0_cause_fixed"):
        if lane.get(key) is not False:
            errors.append(f"manifest lane.{key} must be false")
    if manifest.get("staging", {}).get("require_source_stage_inequality") is not True:
        errors.append("manifest must require source/stage inequality")
    if manifest.get("admin_evidence_before_retry", {}).get("job_id") != "1570434":
        errors.append("administrative evidence gate must remain tied to job 1570434")
    return errors


def path_is_below(path: str, parent: str) -> bool:
    candidate, root = Path(path), Path(parent)
    return candidate.is_absolute() and candidate != root and root in candidate.parents


def validate_observation(observed: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    errors = validate_manifest(manifest)
    provenance = observed.get("provenance", {})
    if provenance.get("branch") != BRANCH:
        errors.append(f"branch mismatch: {provenance.get('branch')!r}")
    if provenance.get("base_commit") != BASE:
        errors.append(f"base commit mismatch: {provenance.get('base_commit')!r}")
    if provenance.get("frozen_files") != FROZEN_FILES:
        errors.append("frozen file hashes do not match")

    metadata = observed.get("metadata")
    if not isinstance(metadata, dict):
        errors.append("full-stage REPRODUCIBILITY_METADATA.json is required")
    else:
        for field in METADATA_FIELDS:
            if field not in metadata:
                errors.append(f"metadata field missing: {field}")
            elif field != "git_status_short" and not metadata[field]:
                errors.append(f"metadata field is empty: {field}")
        fixed = {
            "cwd": SOURCE, "git_commit_full": BASE, "git_commit_short": BASE[:7],
            "git_status_short": "", "source_repo": SOURCE,
        }
        for key, wanted in fixed.items():
            if metadata.get(key) != wanted:
                errors.append(f"metadata {key} is {metadata.get(key)!r}, expected {wanted!r}")
        staged, source = metadata.get("staged_repo", ""), metadata.get("source_repo", "")
        if source == staged:
            errors.append("source and staged repositories must differ")
        if not path_is_below(staged, STAGING_PARENT):
            errors.append(f"staged repository must be an absolute child of {STAGING_PARENT}")

    directives = observed.get("slurm_directives", {})
    for key, wanted in DIRECTIVES.items():
        if directives.get(key) != wanted:
            errors.append(f"Slurm {key} is {directives.get(key)!r}, expected {wanted!r}")
    result_dir = observed.get("result_directory")
    if result_dir != RESULT_DIR or not Path(result_dir or "").is_absolute():
        errors.append(f"result directory must be exact and absolute: {RESULT_DIR}")
    if observed.get("expected_artifacts") != ARTIFACTS:
        errors.append("expected artifact list mismatch")
    guards = observed.get("fail_closed", {})
    for key in ("strict_shell", "requires_metadata", "rejects_equal_paths",
                "critical_exit_unmasked"):
        if guards.get(key) is not True:
            errors.append(f"fail-closed observation is false or missing: {key}")
    return errors


def canonical_observation() -> dict[str, Any]:
    return {
        "provenance": {"branch": BRANCH, "base_commit": BASE,
                       "frozen_files": copy.deepcopy(FROZEN_FILES)},
        "metadata": {
            "command": "fixture-only static launch description",
            "created_at_utc": "2026-08-24T00:00:00+00:00", "cwd": SOURCE,
            "git_commit_full": BASE, "git_commit_short": BASE[:7],
            "git_status_short": "", "hostname": "fixture.invalid",
            "source_repo": SOURCE,
            "staged_repo": STAGING_PARENT + "/fixture-full-stage-code",
        },
        "slurm_directives": copy.deepcopy(DIRECTIVES),
        "result_directory": RESULT_DIR, "expected_artifacts": list(ARTIFACTS),
        "fail_closed": {"strict_shell": True, "requires_metadata": True,
                        "rejects_equal_paths": True, "critical_exit_unmasked": True},
    }


def set_dotted(target: dict[str, Any], dotted: str, value: Any) -> None:
    parts, current = dotted.split("."), target
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            raise ValueError(f"fixture mutation parent is not an object: {dotted}")
        current = child
    if parts[-1] not in current:
        raise ValueError(f"fixture mutation targets unknown field: {dotted}")
    current[parts[-1]] = value


def fixture_result(path: Path, manifest: dict[str, Any]) -> tuple[bool, str, list[str]]:
    fixture = read_json(path)
    label, expected = fixture.get("label"), fixture.get("expected")
    if not isinstance(label, str) or expected not in ("pass", "fail"):
        raise ValueError(f"fixture needs string label and pass/fail expectation: {path}")
    observed = canonical_observation()
    for mutation in fixture.get("mutations", []):
        if not isinstance(mutation, dict) or set(mutation) != {"path", "value"}:
            raise ValueError(f"invalid fixture mutation: {path}")
        set_dotted(observed, mutation["path"], mutation["value"])
    errors = validate_observation(observed, manifest)
    return ("fail" if errors else "pass") == expected, label, errors


def validate_fixtures(directory: Path, manifest: dict[str, Any]) -> int:
    paths = sorted(directory.glob("*.json"))
    if not paths:
        print(f"FAIL fixtures: no JSON fixtures in {directory}", file=sys.stderr)
        return 1
    failed = False
    for path in paths:
        try:
            matched, label, errors = fixture_result(path, manifest)
            expected = read_json(path)["expected"]
        except ValueError as exc:
            matched, label, errors, expected = False, path.name, [str(exc)], "invalid"
        if matched:
            detail = f": {errors[0]}" if errors else ""
            print(f"PASS fixture {label} (expected {expected}){detail}")
        else:
            failed = True
            print(f"FAIL fixture {label} (expected {expected}): {'; '.join(errors)}",
                  file=sys.stderr)
    return int(failed)


def validate_live(repo: Path, manifest: dict[str, Any],
                  allow_owned_dirty: bool = False) -> list[str]:
    errors = validate_manifest(manifest)
    repo = repo.resolve()
    if str(repo) != WORKTREE:
        errors.append(f"live check must run in named worktree {WORKTREE}, got {repo}")
    try:
        if git(repo, "branch", "--show-current") != BRANCH:
            errors.append(f"worktree is not on named branch {BRANCH}")
        run(["git", "merge-base", "--is-ancestor", BASE, "HEAD"], repo)
        changed = set(filter(None, git(repo, "diff", "--name-only", BASE, "HEAD").splitlines()))
        dirty = git(repo, "status", "--porcelain=v1", "--untracked-files=all")
        if dirty and not allow_owned_dirty:
            errors.append("preflight worktree is dirty")
        elif dirty:
            dirty_paths = []
            for line in dirty.splitlines():
                path = line[3:].split(" -> ")[-1]
                if not (path == "notebook.md" or path.startswith(
                        "experiments/phase_a_decode/preflight/")):
                    dirty_paths.append(path)
            if dirty_paths:
                errors.append(
                    f"dirty paths outside development override ownership: {dirty_paths}")
        outside = sorted(path for path in changed if not (
            path == "notebook.md" or path.startswith("experiments/phase_a_decode/preflight/")))
        if outside:
            errors.append(f"changes outside exclusive ownership: {outside}")
    except ValueError as exc:
        errors.append(str(exc))

    source = Path(SOURCE)
    try:
        if git(source, "rev-parse", "HEAD") != BASE:
            errors.append(f"source checkout HEAD is not exact provenance {BASE}")
        if git(source, "status", "--porcelain"):
            errors.append("source checkout is dirty")
    except ValueError as exc:
        errors.append(str(exc))

    for relative, wanted in FROZEN_FILES.items():
        path = repo / relative
        try:
            actual = sha256(path)
        except OSError as exc:
            errors.append(f"cannot hash frozen file {relative}: {exc}")
            continue
        if actual != wanted:
            errors.append(f"frozen file changed: {relative}: {actual}")
        base_bytes = subprocess.run(["git", "show", f"{BASE}:{relative}"], cwd=repo,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    check=False)
        if base_bytes.returncode or hashlib.sha256(base_bytes.stdout).hexdigest() != wanted:
            errors.append(f"base object does not carry frozen hash: {relative}")

    script_path = repo / "experiments/phase_a_decode/run_phase_a.sbatch"
    benchmark_path = repo / "experiments/phase_a_decode/benchmark.py"
    try:
        script, benchmark = script_path.read_text(), benchmark_path.read_text()
    except OSError as exc:
        errors.append(f"cannot read frozen launch files: {exc}")
        return errors
    actual_directives, directive_errors = parse_directives(script)
    errors.extend(directive_errors)
    for key, wanted in DIRECTIVES.items():
        if actual_directives.get(key) != wanted:
            errors.append(f"Slurm {key} is {actual_directives.get(key)!r}, expected {wanted!r}")
    fail_closed = manifest["fail_closed"]
    if fail_closed["strict_shell"] not in script.splitlines():
        errors.append("strict shell mode is missing")
    for fragment in fail_closed["required_script_fragments"]:
        if fragment not in script:
            errors.append(f"required fail-closed script fragment missing: {fragment}")
    commands = [line.strip() for line in script.splitlines() if line.strip()]
    prefix = fail_closed["final_command_prefix"]
    starts = [index for index, line in enumerate(commands) if line.startswith(prefix)]
    if len(starts) != 1 or commands[-1] != '--output "$RESULT_DIR"':
        errors.append("benchmark must remain the final unmasked multi-line command")
    tail = "\n".join(commands[starts[0]:]) if len(starts) == 1 else ""
    if "|| true" in tail or tail.rstrip().endswith("&"):
        errors.append("benchmark exit is masked or asynchronous")
    result_fragment = f'RESULT_DIR="$SOURCE_REPO{RESULT_DIR[len(SOURCE):]}"'
    if result_fragment not in script:
        errors.append("runtime result directory is not the exact absolute-source path")
    combined = script + "\n" + benchmark
    for artifact in ARTIFACTS[1:]:
        if artifact not in combined:
            errors.append(f"expected artifact is not produced by frozen files: {artifact}")
    try:
        run(["bash", "-n", str(script_path)], repo)
    except ValueError as exc:
        errors.append(str(exc))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=HERE.parents[2],
                        help="named preflight worktree root")
    parser.add_argument("--fixtures-dir", type=Path,
                        help="run only labeled JSON fixture validation")
    parser.add_argument("--allow-owned-dirty", action="store_true",
                        help="development only: permit dirt under the two owned paths")
    args = parser.parse_args()
    try:
        manifest = read_json(MANIFEST_PATH)
    except ValueError as exc:
        print(f"FAIL manifest: {exc}", file=sys.stderr)
        return 1
    if args.fixtures_dir:
        return validate_fixtures(args.fixtures_dir, manifest)
    errors = validate_live(args.repo_root, manifest, args.allow_owned_dirty)
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print("PASS live: frozen Phase A contract is internally consistent and submission-free")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
