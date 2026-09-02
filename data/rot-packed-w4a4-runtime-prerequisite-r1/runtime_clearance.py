#!/usr/bin/env python3
"""Fail-closed audit for the one-shot CPU build owner."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys


SCHEMA = "packed-w4a4-cpu-clearance-v1"
TOP_LEVEL_KEYS = {
    "schema_version", "owner", "source", "dependency", "python", "toolchain",
    "helpers", "model", "build", "scheduler", "ledger_path", "terminal_path",
}


class ClearanceError(RuntimeError):
    """The sole CPU owner is not bound to exact reproducible inputs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_regular(path: Path, *, owner_uid: int, mode: int | None = None) -> Path:
    if path.is_symlink():
        raise ClearanceError(f"required control file must not be a symlink: {path}")
    resolved = path.resolve(strict=True)
    status = resolved.stat()
    if not stat.S_ISREG(status.st_mode) or status.st_uid != owner_uid:
        raise ClearanceError(f"required regular file has wrong owner or type: {resolved}")
    observed_mode = stat.S_IMODE(status.st_mode)
    if mode is not None and observed_mode != mode:
        raise ClearanceError(f"mode mismatch for {resolved}: {observed_mode:04o}")
    return resolved


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=root, check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    )
    if completed.returncode:
        raise ClearanceError(f"cannot identify Git HEAD at {root}: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _entry_identity(path: Path) -> dict[str, object]:
    status = path.lstat()
    if stat.S_ISLNK(status.st_mode):
        return {"kind": "symlink", "target": os.readlink(path)}
    if not stat.S_ISREG(status.st_mode):
        raise ClearanceError(f"manifest entry is not a regular file or symlink: {path}")
    if status.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise ClearanceError(f"immutable stage entry is writable: {path}")
    return {
        "kind": "file",
        "sha256": sha256_file(path),
        "executable": bool(status.st_mode & stat.S_IXUSR),
    }


def _verify_manifest(root: Path, manifest_path: Path, expected_sha256: str) -> int:
    if sha256_file(manifest_path) != expected_sha256:
        raise ClearanceError(f"manifest digest mismatch: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("entries")
    if not isinstance(entries, dict) or not entries:
        raise ClearanceError(f"manifest has no entries: {manifest_path}")
    for relative, expected in entries.items():
        path = root / relative
        if _entry_identity(path) != expected:
            raise ClearanceError(f"stage byte identity mismatch: {path}")
    return len(entries)


def _verify_bound_file(record: dict[str, object], owner_uid: int) -> None:
    path = require_regular(Path(str(record["path"])), owner_uid=owner_uid)
    if sha256_file(path) != record["sha256"]:
        raise ClearanceError(f"bound file digest mismatch: {path}")
    if stat.S_IMODE(path.stat().st_mode) != record["mode"]:
        raise ClearanceError(f"bound file mode mismatch: {path}")


def audit(clearance_path: Path, *, require_zero_ledger: bool = True) -> dict[str, object]:
    clearance_path = require_regular(clearance_path, owner_uid=os.getuid(), mode=0o600)
    value = json.loads(clearance_path.read_text(encoding="utf-8"))
    if set(value) != TOP_LEVEL_KEYS or value.get("schema_version") != SCHEMA:
        raise ClearanceError("clearance schema or top-level keys differ")
    owner = value["owner"]
    if owner != {"name": os.environ.get("USER"), "uid": os.getuid()}:
        raise ClearanceError(f"owner identity differs: {owner!r}")
    owner_uid = owner["uid"]

    source = value["source"]
    source_root = Path(source["root"]).resolve(strict=True)
    if not (source_root / ".git").is_dir() or (source_root / ".git").is_symlink():
        raise ClearanceError("project stage lacks independent ordinary Git metadata")
    if _git_head(source_root) != source["head"]:
        raise ClearanceError("project stage HEAD differs")
    source_entries = _verify_manifest(
        source_root, Path(source["manifest_path"]), source["manifest_sha256"],
    )
    metadata_path = require_regular(Path(source["metadata_path"]), owner_uid=owner_uid)
    if sha256_file(metadata_path) != source["metadata_sha256"]:
        raise ClearanceError("project stage metadata digest differs")

    dependency = value["dependency"]
    dependency_root = Path(dependency["root"]).resolve(strict=True)
    cutlass_root = Path(dependency["cutlass_root"]).resolve(strict=True)
    for root, expected in ((dependency_root, dependency["head"]),
                           (cutlass_root, dependency["cutlass_head"])):
        if not (root / ".git").is_dir() or (root / ".git").is_symlink():
            raise ClearanceError(f"dependency lacks independent ordinary Git metadata: {root}")
        if _git_head(root) != expected:
            raise ClearanceError(f"dependency Git HEAD differs: {root}")
    dependency_entries = _verify_manifest(
        dependency_root, Path(dependency["manifest_path"]), dependency["manifest_sha256"],
    )

    _verify_bound_file(value["python"], owner_uid)
    for record in value["toolchain"]:
        _verify_bound_file(record, owner_uid)
    for record in value["helpers"]:
        _verify_bound_file(record, owner_uid)

    model = value["model"]
    snapshot = Path(model["snapshot"]).resolve(strict=True)
    if snapshot.name != model["revision"]:
        raise ClearanceError("model snapshot revision differs")
    for record in model["files"]:
        linked_path = snapshot / record["name"]
        path = linked_path.resolve(strict=True)
        status = path.stat()
        if not stat.S_ISREG(status.st_mode) or status.st_uid != owner_uid:
            raise ClearanceError(f"model file target has wrong owner or type: {path}")
        if str(path) != record["resolved_path"]:
            raise ClearanceError(f"model file target differs: {linked_path}")
        if status.st_size != record["bytes"]:
            raise ClearanceError(f"model file size differs: {path}")
        if record["sha256_verified"] and sha256_file(path) != record["sha256"]:
            raise ClearanceError(f"model file digest differs: {path}")

    build = value["build"]
    build_directory = Path(build["directory"])
    if build_directory.exists():
        raise ClearanceError("accepted build directory must not exist before the sole owner")
    for output_name in ("build_log", "result_json", "validation_log"):
        if Path(build[output_name]).exists():
            raise ClearanceError(f"accepted output already exists: {build[output_name]}")
    builder = Path(build["argv"][0]).resolve(strict=True)
    if builder != Path(value["python"]["path"]).resolve(strict=True):
        raise ClearanceError("builder argv is not pinned to the accepted Python")
    if build["sass_targets"] != ["sm_80", "sm_86"] or build["ptx_targets"] != ["compute_80"]:
        raise ClearanceError("architecture boundary differs from SM80/SM86 plus compute-80 PTX")

    scheduler = value["scheduler"]
    salloc = scheduler["salloc_argv"]
    srun = scheduler["srun_argv"]
    if not salloc or salloc[0] != "/usr/bin/salloc" or not srun or srun[0] != "/usr/bin/srun":
        raise ClearanceError("scheduler executables are not exact")
    if "--pty" not in srun:
        raise ClearanceError("the sole srun must use --pty")
    combined = [argument.lower() for argument in (*salloc, *srun)]
    if any("--gres" in argument or "gpu" in argument for argument in combined):
        raise ClearanceError("CPU build argv must not request or name GPU resources")
    for required in ("--nodes=1", "--ntasks=1"):
        if required not in salloc or required not in srun:
            raise ClearanceError(f"scheduler argv lacks {required}")

    ledger_path = require_regular(Path(value["ledger_path"]), owner_uid=owner_uid, mode=0o600)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    expected_ledger_keys = {
        "owner_attempts", "salloc_attempts", "srun_attempts", "payload_attempts", "terminal_events_fired",
    }
    if set(ledger) != expected_ledger_keys:
        raise ClearanceError("ledger keys differ")
    if require_zero_ledger and any(counter != 0 for counter in ledger.values()):
        raise ClearanceError(f"one-shot ledger is not clear: {ledger!r}")
    if Path(value["terminal_path"]).exists():
        raise ClearanceError("terminal event already exists")

    return {
        "schema_version": "packed-w4a4-cpu-clearance-audit-v1",
        "clearance_path": str(clearance_path),
        "clearance_sha256": sha256_file(clearance_path),
        "source_entries_verified": source_entries,
        "dependency_entries_verified": dependency_entries,
        "toolchain_files_verified": len(value["toolchain"]),
        "helpers_verified": len(value["helpers"]),
        "salloc_argv": salloc,
        "srun_argv": srun,
        "ledger": ledger,
    }


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) != 1:
        raise SystemExit("usage: runtime_clearance.py CLEARANCE.json")
    print(json.dumps(audit(Path(argv[0])), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
