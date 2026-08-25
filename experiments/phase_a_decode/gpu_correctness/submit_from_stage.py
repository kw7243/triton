#!/usr/bin/env python3
"""Verify a full stage, then make the milestone's sole sbatch attempt."""

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_SOURCE = Path(
    "/data/scratch-fast/kwen1/compute-native-vq/worktrees/vq-phase-a-gpu-correctness-r1")
STAGING_PARENT = Path("/data/scratch-fast/kwen1/compute-native-vq/staging")
RESULTS_PARENT = Path("/data/scratch-fast/kwen1/compute-native-vq/results")
BRANCH = "fm/phase-a-gpu-correctness-r1"


def run(args, cwd=None, binary=False):
    return subprocess.check_output(args, cwd=cwd, stderr=subprocess.STDOUT,
                                   text=not binary).strip()


def git(root, *args, binary=False):
    return run(["git", "-C", str(root), *args], binary=binary)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracked_paths(root):
    raw = git(root, "ls-files", "-z", binary=True)
    return [os.fsdecode(item) for item in raw.split(b"\0") if item]


def entry_digest(path):
    if path.is_symlink():
        return hashlib.sha256(b"link\0" + os.fsencode(os.readlink(path))).digest()
    if path.is_file():
        digest = hashlib.sha256()
        digest.update(b"file\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.digest()
    if path.is_dir():
        return hashlib.sha256(b"directory\0").digest()
    raise RuntimeError(f"unsupported tracked entry: {path}")


def aggregate(root, paths):
    digest = hashlib.sha256()
    for relative in paths:
        path = root / relative
        if not os.path.lexists(path):
            raise RuntimeError(f"missing staged input: {relative}")
        digest.update(os.fsencode(relative))
        digest.update(b"\0")
        digest.update(entry_digest(path))
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    result = args.result.resolve(strict=True)
    stage = Path(os.environ["RESEARCH_REPRO_STAGED_DIR"]).resolve(strict=True)
    if source != EXPECTED_SOURCE or result.parent != RESULTS_PARENT or stage.parent != STAGING_PARENT:
        raise SystemExit("source, result, or stage path violates the fixed milestone contract")
    if Path.cwd().resolve() != stage:
        raise SystemExit("submission wrapper must execute from the staged repository")
    if not (stage / ".git").exists() or not (stage / "REPRODUCIBILITY_METADATA.json").is_file():
        raise SystemExit("stage is missing .git or reproducibility metadata")

    source_status = git(source, "status", "--porcelain=v1")
    if source_status:
        raise SystemExit("source became dirty before staging: " + source_status.replace("\n", "; "))
    if git(source, "branch", "--show-current") != BRANCH:
        raise SystemExit("source branch changed before staging")
    if Path(git(stage, "rev-parse", "--show-toplevel")) != stage:
        raise SystemExit("staged .git does not resolve to the staged repository")

    metadata_path = stage / "REPRODUCIBILITY_METADATA.json"
    metadata = json.loads(metadata_path.read_text())
    source_head, stage_head = git(source, "rev-parse", "HEAD"), git(stage, "rev-parse", "HEAD")
    source_tree, stage_tree = (git(root, "rev-parse", "HEAD^{tree}") for root in (source, stage))
    if metadata.get("source_repo") != str(source) or metadata.get("staged_repo") != str(stage):
        raise SystemExit("reproducibility metadata paths do not match source and stage")
    if metadata.get("git_commit_full") != source_head or metadata.get("git_status_short"):
        raise SystemExit("reproducibility metadata does not describe a clean source HEAD")
    if source_head != stage_head or source_tree != stage_tree:
        raise SystemExit("source/stage commit or tree mismatch")

    source_paths, stage_paths = tracked_paths(source), tracked_paths(stage)
    if source_paths != stage_paths:
        raise SystemExit("source/stage tracked path lists differ")
    source_hash, stage_hash = aggregate(source, source_paths), aggregate(stage, stage_paths)
    if source_hash != stage_hash:
        raise SystemExit("source/stage tracked content hashes differ")
    git_pointer_hash = sha256(source / ".git")
    if sha256(stage / ".git") != git_pointer_hash:
        raise SystemExit("source/stage .git bytes differ")

    sbatch = [
        "sbatch", "--parsable", "--account=vision-torralba-urops-meng",
        "--qos=vision-torralba-interactive", "--partition=vision-torralba-rtx3090",
        "--nodes=1", "--ntasks=1", "--cpus-per-task=4", "--gres=gpu:1",
        "--mem=16G", "--time=00:15:00",
        f"--output={result}/slurm-%j.out", f"--error={result}/slurm-%j.out",
        f"--export=ALL,SOURCE_REPO={source},RESULT_DIR={result}",
        "experiments/phase_a_decode/gpu_correctness/run_gpu_correctness.sbatch",
    ]
    proof = {
        "schema": "vq-phase-a-gpu-correctness-r1-stage/v1",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_repo": str(source), "staged_repo": str(stage), "result_root": str(result),
        "branch": BRANCH, "commit": source_head, "tree": source_tree,
        "source_status": source_status, "tracked_files": len(source_paths),
        "tracked_content_sha256": source_hash, "git_pointer_sha256": git_pointer_hash,
        "reproducibility_metadata_sha256": sha256(metadata_path),
        "sbatch_argv": sbatch,
    }
    (result / "stage_verification.json").write_text(json.dumps(proof, indent=2) + "\n")
    (result / "submission_request.txt").write_text(shlex.join(sbatch) + "\n")

    response = subprocess.run(sbatch, cwd=stage, text=True, capture_output=True)
    response_record = {"returncode": response.returncode, "stdout": response.stdout.strip(),
                       "stderr": response.stderr.strip()}
    (result / "submission_response.json").write_text(json.dumps(response_record, indent=2) + "\n")
    if response.returncode != 0:
        raise SystemExit(response.returncode)
    job_id = response.stdout.strip()
    if re.fullmatch(r"[0-9]+", job_id) is None:
        raise SystemExit("sbatch returned a non-numeric response")
    print(job_id)


if __name__ == "__main__":
    main()
