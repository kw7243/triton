#!/usr/bin/env python3
"""Independent, GPU-free audit of the corrected Phase A repository stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess


EXPECTED_HEAD = "48a972220979197359a324ab102eb8de24ce321f"
EXPECTED_TREE = "ba5f873fb9c2defa7b3af15b971cf5fb8d3092fb"
METADATA_NAME = "REPRODUCIBILITY_METADATA.json"
MANIFEST_NAME = "REPRODUCIBILITY_MANIFEST.json"
CONTROL_NAMES = {METADATA_NAME, MANIFEST_NAME}


def run(*argv: str, cwd: Path) -> str:
    return subprocess.check_output(argv, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def regular_bytes(path: Path) -> bytes:
    mode = path.lstat().st_mode
    if not stat.S_ISREG(mode) or path.is_symlink():
        raise RuntimeError(f"not a regular non-symlink file: {path}")
    return path.read_bytes()


def identity(path: Path) -> dict[str, object]:
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode):
        return {"kind": "symlink", "target": os.readlink(path)}
    if not stat.S_ISREG(mode):
        raise RuntimeError(f"unsupported stage entry: {path}")
    return {
        "kind": "file",
        "sha256": sha256(path.read_bytes()),
        "executable": bool(mode & stat.S_IXUSR),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage")
    parser.add_argument("output")
    args = parser.parse_args()
    stage = Path(args.stage).resolve(strict=True)
    output = Path(args.output)
    if output.exists():
        raise RuntimeError(f"audit output already exists: {output}")
    if Path(run("git", "rev-parse", "--show-toplevel", cwd=stage)).resolve() != stage:
        raise RuntimeError("stage is not the exact Git top level")
    if not (stage / ".git").is_dir() or (stage / ".git").is_symlink():
        raise RuntimeError("stage .git is not an ordinary directory")
    if run("git", "rev-parse", "--git-dir", cwd=stage) != ".git":
        raise RuntimeError("stage git-dir is not local")
    if run("git", "rev-parse", "--git-common-dir", cwd=stage) != ".git":
        raise RuntimeError("stage common-dir is not local")
    if run("git", "rev-parse", "--is-shallow-repository", cwd=stage) != "false":
        raise RuntimeError("stage is shallow")
    if subprocess.run(
        ("git", "config", "--get-regexp", r"^remote\..*\.promisor$"),
        cwd=stage,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    ).returncode == 0:
        raise RuntimeError("stage has a promisor remote")
    alternates = stage / ".git" / "objects" / "info" / "alternates"
    if alternates.exists() and alternates.read_text(encoding="utf-8").strip():
        raise RuntimeError("stage uses object alternates")
    head = run("git", "rev-parse", "--verify", "HEAD^{commit}", cwd=stage)
    tree = run("git", "rev-parse", "HEAD^{tree}", cwd=stage)
    if head != EXPECTED_HEAD or tree != EXPECTED_TREE:
        raise RuntimeError(f"unexpected stage identity: {head} {tree}")
    run("git", "cat-file", "-e", f"{head}^{{commit}}", cwd=stage)
    run("git", "fsck", "--connectivity-only", "--no-dangling", cwd=stage)

    metadata_bytes = regular_bytes(stage / METADATA_NAME)
    manifest_bytes = regular_bytes(stage / MANIFEST_NAME)
    metadata = json.loads(metadata_bytes)
    manifest = json.loads(manifest_bytes)
    entries = manifest["entries"]
    if metadata["schema_version"] != "phase-a-repository-stage-v3":
        raise RuntimeError("wrong stage metadata schema")
    if manifest["schema_version"] != "phase-a-working-tree-manifest-v1":
        raise RuntimeError("wrong stage manifest schema")
    if metadata["source_head"] != head:
        raise RuntimeError("stage metadata source head mismatch")
    if metadata["staged_git_common_dir"] != str(stage / ".git"):
        raise RuntimeError("stage metadata common-dir mismatch")
    if metadata["tracked_and_untracked_entries"] != len(entries):
        raise RuntimeError("stage metadata entry count mismatch")
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    if sha256(canonical) != metadata["working_tree_manifest_sha256"]:
        raise RuntimeError("canonical manifest digest mismatch")
    if sha256(manifest_bytes) != metadata["working_tree_manifest_file_sha256"]:
        raise RuntimeError("manifest file digest mismatch")

    tracked = set(filter(None, run("git", "ls-files", "-z", cwd=stage).split("\0")))
    if set(entries) != tracked:
        raise RuntimeError("clean-source stage manifest differs from tracked paths")
    observed = {name: identity(stage / name) for name in entries}
    if observed != entries:
        raise RuntimeError("stage file identities differ from manifest")
    status = set(filter(None, run("git", "status", "--porcelain=v1", "--untracked-files=all", cwd=stage).splitlines()))
    if status != {f"?? {METADATA_NAME}", f"?? {MANIFEST_NAME}"}:
        raise RuntimeError(f"unexpected stage status: {sorted(status)}")

    visible_files: set[str] = set()
    for root, directories, files in os.walk(stage, followlinks=False):
        root_path = Path(root)
        if root_path == stage:
            directories[:] = [name for name in directories if name != ".git"]
        for name in files:
            visible_files.add((root_path / name).relative_to(stage).as_posix())
        for name in directories:
            candidate = root_path / name
            if candidate.is_symlink():
                visible_files.add(candidate.relative_to(stage).as_posix())
    if visible_files != set(entries) | CONTROL_NAMES:
        raise RuntimeError("stage filesystem contains an undeclared non-Git input")

    key_paths = [
        "experiments/structured_hadamard/phase_a/activation_quantizer.py",
        "experiments/structured_hadamard/phase_a/execute.py",
        "experiments/structured_hadamard/phase_a/stage_repository.py",
    ]
    key_files = {}
    for name in key_paths:
        data = regular_bytes(stage / name)
        git_data = subprocess.check_output(("git", "show", f"HEAD:{name}"), cwd=stage)
        if data != git_data:
            raise RuntimeError(f"staged bytes do not match Git: {name}")
        key_files[name] = {"size": len(data), "sha256": sha256(data), "mode": oct((stage / name).stat().st_mode & 0o777)}

    result = {
        "schema_version": "rot-phasea-corrected-stage-audit-v1",
        "stage": str(stage),
        "head": head,
        "tree": tree,
        "git_dir": str(stage / ".git"),
        "entry_count": len(entries),
        "manifest_sha256": metadata["working_tree_manifest_sha256"],
        "manifest_file_sha256": sha256(manifest_bytes),
        "metadata_sha256": sha256(metadata_bytes),
        "status": sorted(status),
        "key_files": key_files,
    }
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
