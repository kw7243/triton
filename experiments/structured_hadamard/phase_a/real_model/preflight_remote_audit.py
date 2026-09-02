"""Fail-closed remote preflight for the decisive real-model Phase A run.

This helper is intentionally standard-library only.  It consumes one exact
JSON clearance file and audits files, Git repositories, package metadata,
argv, and one-shot state without importing any scientific package.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Iterable, Mapping, Sequence


SCHEMA_VERSION = "phase-a-real-model-remote-preflight-v1"
RESULT_SCHEMA_VERSION = "phase-a-real-model-remote-preflight-result-v1"
TOP_LEVEL_KEYS = {
    "schema_version",
    "owner",
    "project",
    "accepted_project",
    "dependency",
    "extension",
    "model",
    "data",
    "environment",
    "helpers",
    "execution",
    "ledger",
    "terminal_path",
    "audit_output",
}


class AuditError(RuntimeError):
    """The remote state differs from the pinned preflight contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_regular(path: Path, *, owner_uid: int, mode: int | None = None) -> Path:
    if path.is_symlink():
        raise AuditError(f"required control file must not be a symlink: {path}")
    resolved = path.resolve(strict=True)
    status = resolved.stat()
    if not stat.S_ISREG(status.st_mode) or status.st_uid != owner_uid:
        raise AuditError(f"required regular file has wrong type or owner: {resolved}")
    observed_mode = stat.S_IMODE(status.st_mode)
    if mode is not None and observed_mode != mode:
        raise AuditError(f"mode mismatch for {resolved}: {observed_mode:04o}")
    return resolved


def _run(argv: Sequence[str], *, cwd: Path) -> str:
    completed = subprocess.run(
        tuple(argv), cwd=cwd, check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    )
    if completed.returncode:
        raise AuditError(
            f"command failed ({completed.returncode}): {list(argv)!r}: {completed.stderr.strip()}"
        )
    return completed.stdout


def _git(root: Path, *argv: str) -> str:
    return _run(("git", *argv), cwd=root).strip()


def _entry_identity(path: Path, *, owner_uid: int, immutable: bool) -> dict[str, object]:
    status = path.lstat()
    if status.st_uid != owner_uid:
        raise AuditError(f"manifest entry has wrong owner: {path}")
    if stat.S_ISLNK(status.st_mode):
        return {"kind": "symlink", "target": os.readlink(path)}
    if not stat.S_ISREG(status.st_mode):
        raise AuditError(f"manifest entry is not regular or a symlink: {path}")
    if immutable and status.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise AuditError(f"immutable manifest entry is writable: {path}")
    return {
        "kind": "file",
        "sha256": sha256_file(path),
        "executable": bool(status.st_mode & stat.S_IXUSR),
    }


def _manifest_entries(payload: object) -> Mapping[str, Mapping[str, object]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("entries"), dict):
        raise AuditError("manifest must contain an entries object")
    entries = payload["entries"]
    if not entries:
        raise AuditError("manifest entries must not be empty")
    for relative, identity in entries.items():
        path = Path(relative)
        if (not isinstance(relative, str) or path.is_absolute() or ".." in path.parts
                or not isinstance(identity, dict)):
            raise AuditError(f"unsafe or malformed manifest entry: {relative!r}")
    return entries


def audit_manifest(root: Path, record: Mapping[str, object], *, owner_uid: int) -> dict[str, object]:
    manifest = require_regular(Path(str(record["manifest_path"])), owner_uid=owner_uid)
    if sha256_file(manifest) != record["manifest_file_sha256"]:
        raise AuditError(f"manifest file digest mismatch: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    entries = _manifest_entries(payload)
    if canonical_sha256(entries) != record["entries_sha256"]:
        raise AuditError(f"canonical manifest digest mismatch: {manifest}")
    immutable = bool(record["immutable"])
    observed = {
        relative: _entry_identity(root / relative, owner_uid=owner_uid, immutable=immutable)
        for relative in entries
    }
    if observed != entries:
        raise AuditError(f"manifest content mismatch beneath {root}")
    return {
        "path": str(manifest),
        "file_sha256": record["manifest_file_sha256"],
        "entries_sha256": record["entries_sha256"],
        "entries_verified": len(entries),
        "immutable": immutable,
    }


def audit_git_stage(record: Mapping[str, object], *, owner_uid: int) -> dict[str, object]:
    root = Path(str(record["root"])).resolve(strict=True)
    if not root.is_dir() or root.stat().st_uid != owner_uid:
        raise AuditError(f"stage root has wrong type or owner: {root}")
    if not (root / ".git").is_dir() or (root / ".git").is_symlink():
        raise AuditError(f"stage lacks ordinary independent Git metadata: {root}")
    git_dir = Path(_git(root, "rev-parse", "--git-dir"))
    common_dir = Path(_git(root, "rev-parse", "--git-common-dir"))
    git_dir = (root / git_dir).resolve() if not git_dir.is_absolute() else git_dir.resolve()
    common_dir = (root / common_dir).resolve() if not common_dir.is_absolute() else common_dir.resolve()
    if git_dir != root / ".git" or common_dir != root / ".git":
        raise AuditError(f"stage Git metadata escapes its root: {root}")
    alternates = root / ".git" / "objects" / "info" / "alternates"
    if alternates.exists() and alternates.read_text(encoding="utf-8").strip():
        raise AuditError(f"stage uses external Git object alternates: {root}")
    if _git(root, "rev-parse", "--is-shallow-repository") != "false":
        raise AuditError(f"stage is shallow: {root}")
    promisor = subprocess.run(
        ("git", "config", "--get-regexp", r"^remote\..*\.promisor$"), cwd=root,
        check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    )
    if promisor.returncode == 0 and promisor.stdout.strip():
        raise AuditError(f"stage is a promisor repository: {root}")
    head = _git(root, "rev-parse", "HEAD^{commit}")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    if head != record["head"] or tree != record["tree"]:
        raise AuditError(f"stage Git identity mismatch: {root}")
    required_ancestor = str(record["required_ancestor"])
    ancestor = subprocess.run(
        ("git", "merge-base", "--is-ancestor", required_ancestor, head), cwd=root,
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    )
    if ancestor.returncode:
        raise AuditError(f"required ancestor {required_ancestor} is absent from {root}")
    _git(root, "fsck", "--connectivity-only", "--no-dangling")
    manifest_result = audit_manifest(root, record["manifest"], owner_uid=owner_uid)
    metadata = require_regular(Path(str(record["metadata_path"])), owner_uid=owner_uid)
    if sha256_file(metadata) != record["metadata_sha256"]:
        raise AuditError(f"stage metadata digest mismatch: {metadata}")
    if bool(record["manifest"]["immutable"]):
        for control in (manifest_result["path"], str(metadata)):
            if Path(control).stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
                raise AuditError(f"immutable stage control file is writable: {control}")
    return {
        "root": str(root), "head": head, "tree": tree,
        "required_ancestor": required_ancestor, "manifest": manifest_result,
        "metadata_sha256": record["metadata_sha256"],
    }


def audit_dependency(record: Mapping[str, object], *, owner_uid: int) -> dict[str, object]:
    root = Path(str(record["root"])).resolve(strict=True)
    cutlass = Path(str(record["cutlass_root"])).resolve(strict=True)
    for candidate, expected in ((root, record["head"]), (cutlass, record["cutlass_head"])):
        if not (candidate / ".git").is_dir() or (candidate / ".git").is_symlink():
            raise AuditError(f"dependency lacks ordinary Git metadata: {candidate}")
        if _git(candidate, "rev-parse", "HEAD^{commit}") != expected:
            raise AuditError(f"dependency commit mismatch: {candidate}")
        _git(candidate, "fsck", "--connectivity-only", "--no-dangling")
    manifest = audit_manifest(root, record["manifest"], owner_uid=owner_uid)
    return {
        "root": str(root), "head": record["head"],
        "cutlass_root": str(cutlass), "cutlass_head": record["cutlass_head"],
        "manifest": manifest,
    }


def _audit_file_record(root: Path, record: Mapping[str, object], *, owner_uid: int) -> dict[str, object]:
    linked = root / str(record["name"])
    resolved = linked.resolve(strict=True)
    status = resolved.stat()
    if not stat.S_ISREG(status.st_mode) or status.st_uid != owner_uid:
        raise AuditError(f"data file has wrong type or owner: {resolved}")
    if str(resolved) != record["resolved_path"]:
        raise AuditError(f"resolved data path differs: {linked}")
    if status.st_size != record["bytes"] or sha256_file(resolved) != record["sha256"]:
        raise AuditError(f"data file identity differs: {resolved}")
    return {
        "name": record["name"], "resolved_path": str(resolved),
        "bytes": status.st_size, "sha256": record["sha256"],
    }


def audit_cache(record: Mapping[str, object], *, owner_uid: int, kind: str) -> dict[str, object]:
    root = Path(str(record["root"])).resolve(strict=True)
    if root.name != record["revision"] or not root.is_dir() or root.stat().st_uid != owner_uid:
        raise AuditError(f"{kind} cache root identity differs: {root}")
    files = [_audit_file_record(root, item, owner_uid=owner_uid) for item in record["files"]]
    if not files:
        raise AuditError(f"{kind} cache has no bound files")
    return {
        "name": record["name"], "configuration": record.get("configuration"),
        "revision": record["revision"], "root": str(root), "files": files,
    }


def audit_extension(record: Mapping[str, object], *, owner_uid: int) -> dict[str, object]:
    extension = require_regular(Path(str(record["path"])), owner_uid=owner_uid, mode=int(record["mode"]))
    if extension.stat().st_size != record["bytes"] or sha256_file(extension) != record["sha256"]:
        raise AuditError(f"accepted extension identity differs: {extension}")
    tool = require_regular(Path(str(record["cuobjdump_path"])), owner_uid=owner_uid,
                           mode=int(record["cuobjdump_mode"]))
    if sha256_file(tool) != record["cuobjdump_sha256"]:
        raise AuditError(f"cuobjdump identity differs: {tool}")
    elf = _run((str(tool), "--list-elf", str(extension)), cwd=extension.parent)
    ptx = _run((str(tool), "--dump-ptx", str(extension)), cwd=extension.parent)
    for target in record["sass_targets"]:
        if target not in elf:
            raise AuditError(f"accepted extension lacks native target {target}")
    for target in record["ptx_markers"]:
        if target not in ptx:
            raise AuditError(f"accepted extension lacks PTX marker {target}")
    for forbidden in record["forbidden_arch_markers"]:
        if forbidden in elf:
            raise AuditError(f"accepted extension unexpectedly contains {forbidden}")
    return {
        "path": str(extension), "bytes": extension.stat().st_size,
        "mode": stat.S_IMODE(extension.stat().st_mode), "sha256": record["sha256"],
        "sass_targets": list(record["sass_targets"]),
        "ptx_markers": list(record["ptx_markers"]),
        "cuobjdump_path": str(tool), "cuobjdump_sha256": record["cuobjdump_sha256"],
    }


def _normalize_package_name(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def _distribution_record(distribution: importlib_metadata.Distribution, *, exact_bytes: bool,
                         allowed_roots: Iterable[Path]) -> dict[str, object]:
    roots = tuple(path.resolve(strict=True) for path in allowed_roots)
    metadata_name = distribution.metadata.get("Name")
    if not metadata_name:
        raise AuditError("installed distribution lacks a Name field")
    files = distribution.files
    if files is None:
        raise AuditError(f"distribution lacks an installed file record: {metadata_name}")
    byte_records = []
    record_files = []
    for item in sorted(files, key=lambda value: str(value)):
        path = Path(distribution.locate_file(item)).resolve(strict=True)
        if not any(path == root or root in path.parents for root in roots):
            raise AuditError(f"package file escapes accepted environment roots: {path}")
        status = path.stat()
        if not stat.S_ISREG(status.st_mode):
            raise AuditError(f"package record is not a regular file: {path}")
        relative_root = next(root for root in roots if path == root or root in path.parents)
        relative = path.relative_to(relative_root).as_posix()
        if relative.endswith(".dist-info/RECORD"):
            record_files.append({"path": str(path), "sha256": sha256_file(path)})
        if exact_bytes:
            byte_records.append({
                "root": str(relative_root), "path": relative,
                "bytes": status.st_size, "sha256": sha256_file(path),
            })
    if not record_files:
        raise AuditError(f"distribution lacks a RECORD file: {metadata_name}")
    result = {
        "name": metadata_name,
        "version": distribution.version,
        "record_files": record_files,
        "installed_files": len(files),
        "exact_bytes": exact_bytes,
    }
    if exact_bytes:
        result["byte_manifest_sha256"] = canonical_sha256(byte_records)
        result["bytes_hashed"] = sum(item["bytes"] for item in byte_records)
        result["files_hashed"] = len(byte_records)
    return result


def audit_environment(record: Mapping[str, object], *, owner_uid: int) -> dict[str, object]:
    python_record = record["python"]
    python = require_regular(Path(str(python_record["path"])), owner_uid=owner_uid,
                             mode=int(python_record["mode"]))
    if sha256_file(python) != python_record["sha256"]:
        raise AuditError(f"Python interpreter digest differs: {python}")
    site_paths = [Path(str(path)).resolve(strict=True) for path in record["site_paths"]]
    allowed_roots = [Path(str(path)).resolve(strict=True) for path in record["allowed_roots"]]
    for path in site_paths:
        if not path.is_dir() or path.stat().st_uid != owner_uid:
            raise AuditError(f"site-packages path has wrong type or owner: {path}")
    distributions: dict[str, list[importlib_metadata.Distribution]] = {}
    for distribution in importlib_metadata.distributions(path=[str(path) for path in site_paths]):
        name = distribution.metadata.get("Name")
        if name:
            distributions.setdefault(_normalize_package_name(name), []).append(distribution)
    package_results = []
    for expected in record["packages"]:
        normalized = _normalize_package_name(str(expected["name"]))
        matches = distributions.get(normalized, [])
        if len(matches) != 1:
            raise AuditError(f"expected exactly one installed {expected['name']}, found {len(matches)}")
        distribution = matches[0]
        if distribution.version != expected["version"]:
            raise AuditError(
                f"package version differs for {expected['name']}: {distribution.version}"
            )
        observed = _distribution_record(
            distribution, exact_bytes=bool(expected["exact_bytes"]), allowed_roots=allowed_roots,
        )
        if "byte_manifest_sha256" in expected:
            if observed.get("byte_manifest_sha256") != expected["byte_manifest_sha256"]:
                raise AuditError(f"package byte manifest differs for {expected['name']}")
        package_results.append(observed)
    metadata_inventory = sorted(
        (distribution.metadata.get("Name"), distribution.version)
        for values in distributions.values() for distribution in values
        if distribution.metadata.get("Name")
    )
    runtime_env = record["runtime_env"]
    if (not isinstance(runtime_env, dict)
            or any(not isinstance(key, str) or not isinstance(value, str)
                   for key, value in runtime_env.items())):
        raise AuditError("runtime_env must be a string mapping")
    if any(any(fragment in key.upper() for fragment in ("TOKEN", "SECRET", "PASSWORD", "KEY"))
           for key in runtime_env):
        raise AuditError("runtime_env must not bind credential-like keys")
    expected_pythonpath = os.pathsep.join(str(path) for path in site_paths)
    if runtime_env.get("PYTHONPATH") != expected_pythonpath:
        raise AuditError("runtime PYTHONPATH differs from the audited site-packages order")
    return {
        "python": {
            "path": str(python), "sha256": python_record["sha256"],
            "mode": stat.S_IMODE(python.stat().st_mode), "version": sys.version,
        },
        "site_paths": [str(path) for path in site_paths],
        "metadata_inventory_sha256": canonical_sha256(metadata_inventory),
        "metadata_inventory_count": len(metadata_inventory),
        "packages": package_results,
        "runtime_env": runtime_env,
        "scientific_packages_imported": False,
    }


def audit_helpers(records: Sequence[Mapping[str, object]], *, owner_uid: int) -> list[dict[str, object]]:
    results = []
    for record in records:
        path = require_regular(Path(str(record["path"])), owner_uid=owner_uid,
                               mode=int(record["mode"]))
        if sha256_file(path) != record["sha256"]:
            raise AuditError(f"helper digest differs: {path}")
        results.append({
            "name": record["name"], "path": str(path), "sha256": record["sha256"],
            "mode": stat.S_IMODE(path.stat().st_mode),
        })
    return results


def _audit_argv(argv: object, *, label: str) -> list[str]:
    if not isinstance(argv, list) or not argv or any(not isinstance(value, str) or not value for value in argv):
        raise AuditError(f"{label} must be a nonempty string argv")
    forbidden = {"sh", "bash", "-c", "--wrap", "sbatch"}
    if any(Path(value).name in forbidden or value in forbidden for value in argv):
        raise AuditError(f"{label} contains a shell/submission boundary")
    return list(argv)


def audit_execution(record: Mapping[str, object], *, owner_uid: int) -> dict[str, object]:
    owner = _audit_argv(record["owner_argv"], label="owner_argv")
    allocated = _audit_argv(record["allocated_argv"], label="allocated_argv")
    driver = _audit_argv(record["driver_argv"], label="driver_argv")
    salloc = _audit_argv(record["salloc_argv"], label="salloc_argv")
    srun = _audit_argv(record["srun_argv"], label="srun_argv")
    if salloc[0] != "/usr/bin/salloc" or srun[0] != "/usr/bin/srun" or "--pty" not in srun:
        raise AuditError("scheduler argv must be exact salloc then srun --pty")
    if record["argv_sha256"] != canonical_sha256({
            "owner_argv": owner, "allocated_argv": allocated, "driver_argv": driver,
            "salloc_argv": salloc, "srun_argv": srun}):
        raise AuditError("exact execution argv digest differs")
    output_parent = Path(str(record["output_parent"])).resolve(strict=True)
    status = output_parent.stat()
    if not stat.S_ISDIR(status.st_mode) or status.st_uid != owner_uid:
        raise AuditError(f"output parent has wrong type or owner: {output_parent}")
    if stat.S_IMODE(status.st_mode) != record["output_parent_mode"]:
        raise AuditError(f"output parent mode differs: {output_parent}")
    if not os.access(output_parent, os.W_OK | os.X_OK):
        raise AuditError(f"output parent is not writable/searchable: {output_parent}")
    output = Path(str(record["output_directory"])).resolve()
    if output.parent != output_parent or output.exists():
        raise AuditError(f"output directory exists or escapes its parent: {output}")
    return {
        "owner_argv": owner, "allocated_argv": allocated, "driver_argv": driver,
        "salloc_argv": salloc, "srun_argv": srun,
        "argv_sha256": record["argv_sha256"], "duration_estimate": record["duration_estimate"],
        "output_parent": str(output_parent), "output_parent_mode": record["output_parent_mode"],
        "output_directory": str(output),
    }


def audit(clearance_path: Path, *, expected_clearance_sha256: str,
          expected_helper_sha256: str, allow_existing_output: bool = False) -> dict[str, object]:
    helper = Path(__file__).resolve(strict=True)
    if sha256_file(helper) != expected_helper_sha256:
        raise AuditError(f"executed helper digest differs: {helper}")
    clearance_path = require_regular(clearance_path, owner_uid=os.getuid(), mode=0o600)
    if sha256_file(clearance_path) != expected_clearance_sha256:
        raise AuditError(f"clearance digest differs: {clearance_path}")
    value = json.loads(clearance_path.read_text(encoding="utf-8"))
    if set(value) != TOP_LEVEL_KEYS or value.get("schema_version") != SCHEMA_VERSION:
        raise AuditError("clearance schema or top-level keys differ")
    owner = value["owner"]
    if owner != {"name": os.environ.get("USER"), "uid": os.getuid()}:
        raise AuditError(f"owner identity differs: {owner!r}")
    owner_uid = int(owner["uid"])
    ledger_path = require_regular(Path(str(value["ledger"]["path"])), owner_uid=owner_uid, mode=0o600)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger != value["ledger"]["expected_zero"]:
        raise AuditError(f"one-shot owner ledger is not clear: {ledger!r}")
    status_path = require_regular(
        Path(str(value["ledger"]["status_path"])), owner_uid=owner_uid, mode=0o600,
    )
    if sha256_file(status_path) != value["ledger"]["status_sha256"]:
        raise AuditError(f"task status digest differs: {status_path}")
    terminal = Path(str(value["terminal_path"]))
    if terminal.exists():
        raise AuditError(f"terminal event already exists: {terminal}")
    output = Path(str(value["audit_output"])).resolve()
    if output.exists() and not allow_existing_output:
        raise AuditError(f"audit output already exists: {output}")
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "passed",
        "helper": {"path": str(helper), "sha256": expected_helper_sha256},
        "clearance": {"path": str(clearance_path), "sha256": expected_clearance_sha256},
        "owner": owner,
        "project": audit_git_stage(value["project"], owner_uid=owner_uid),
        "accepted_project": audit_git_stage(value["accepted_project"], owner_uid=owner_uid),
        "dependency": audit_dependency(value["dependency"], owner_uid=owner_uid),
        "extension": audit_extension(value["extension"], owner_uid=owner_uid),
        "model": audit_cache(value["model"], owner_uid=owner_uid, kind="model"),
        "data": audit_cache(value["data"], owner_uid=owner_uid, kind="data"),
        "environment": audit_environment(value["environment"], owner_uid=owner_uid),
        "helpers": audit_helpers(value["helpers"], owner_uid=owner_uid),
        "execution": audit_execution(value["execution"], owner_uid=owner_uid),
        "ledger": {"path": str(ledger_path), "value": ledger,
                   "status_path": str(status_path), "status_sha256": value["ledger"]["status_sha256"]},
        "terminal_path": str(terminal),
    }
    return result


def _exclusive_json(path: Path, value: object) -> None:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clearance", type=Path, required=True)
    parser.add_argument("--expected-clearance-sha256", required=True)
    parser.add_argument("--expected-helper-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = audit(
            args.clearance, expected_clearance_sha256=args.expected_clearance_sha256,
            expected_helper_sha256=args.expected_helper_sha256,
        )
        expected_output = Path(json.loads(args.clearance.read_text(encoding="utf-8"))["audit_output"])
        if args.output.resolve() != expected_output.resolve():
            raise AuditError("output argv differs from the clearance-bound audit output")
        _exclusive_json(args.output, result)
    except (AuditError, json.JSONDecodeError, KeyError, OSError, TypeError, ValueError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
