"""Fail-closed full-repository staging for future experimental commands.

The staged checkout has an ordinary, self-contained ``.git`` directory even
when the source is a linked worktree. The source ``HEAD`` is cloned without
local hardlinks, then tracked and all included untracked working-tree entries
(including Git-ignored inputs) are copied and verified before an optional
command may run. Only the explicit generated/cache/virtualenv directory trees
below are excluded by default; tracked paths always override those exclusions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from typing import Sequence


# Root-level generated trees are excluded recursively. Cache and virtualenv
# directory names are excluded at any depth. These are the only default
# working-tree exclusions; .git metadata is handled independently by cloning.
DEFAULT_EXCLUDED_ROOT_DIRECTORIES = (
    "staging",
    "out",
    "outputs",
    "eval_outputs",
    "slurm_outputs",
    "wandb",
)
DEFAULT_EXCLUDED_DIRECTORY_NAMES = (
    ".cache",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
)
STAGE_METADATA_NAME = "REPRODUCIBILITY_METADATA.json"
STAGE_MANIFEST_NAME = "REPRODUCIBILITY_MANIFEST.json"


class StageError(RuntimeError):
    """The helper could not prove that a stage is complete and independent."""


def _run(args: Sequence[str], *, cwd: Path, check: bool = True, text: bool = True):
    completed = subprocess.run(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=text,
                               check=False)
    if check and completed.returncode:
        stderr = completed.stderr.strip() if text else completed.stderr.decode(errors="replace").strip()
        raise StageError(f"command failed ({completed.returncode}): {' '.join(args)}: {stderr}")
    return completed


def _git(root: Path, *args: str, text: bool = True):
    return _run(("git", *args), cwd=root, text=text).stdout


def _absolute_git_path(root: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _validate_source(source: Path) -> tuple[Path, str, Path]:
    source = source.resolve(strict=True)
    if not source.is_dir():
        raise StageError("source must be a repository worktree directory")
    top = Path(_git(source, "rev-parse", "--show-toplevel").strip()).resolve()
    if top != source:
        raise StageError(f"source must be the exact repository top level: {top}")
    if _git(source, "rev-parse", "--is-bare-repository").strip() != "false":
        raise StageError("source must be a non-bare worktree")
    if _git(source, "rev-parse", "--is-shallow-repository").strip() != "false":
        raise StageError("source must not be shallow")
    promisor = _run(("git", "config", "--get-regexp", r"^remote\..*\.promisor$"), cwd=source, check=False)
    if promisor.returncode == 0 and promisor.stdout.strip():
        raise StageError("partial/promisor repositories cannot prove a complete object database")
    head = _git(source, "rev-parse", "--verify", "HEAD^{commit}").strip()
    if len(head) != 40:
        raise StageError("source HEAD did not resolve to a full commit id")
    if _git(source, "ls-files", "-u", "-z", text=False):
        raise StageError("source has unmerged index entries")
    staged_entries = _git(source, "ls-files", "--stage", "-z", text=False).split(b"\0")
    if any(entry.startswith(b"160000 ") for entry in staged_entries if entry):
        raise StageError("submodules are unsupported because their independent metadata cannot be proven")
    _git(source, "fsck", "--connectivity-only", "--no-dangling")
    common_dir = _absolute_git_path(source, _git(source, "rev-parse", "--git-common-dir").strip())
    return source, head, common_dir


def _source_paths(source: Path) -> tuple[Path, ...]:
    tracked_output = _git(source, "ls-files", "-z", "--cached", text=False)
    tracked_paths = set()
    for raw in tracked_output.split(b"\0"):
        if not raw:
            continue
        relative = Path(os.fsdecode(raw))
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] == ".git":
            raise StageError(f"unsafe repository path: {relative}")
        if relative in tracked_paths:
            raise StageError("Git returned duplicate tracked paths")
        tracked_paths.add(relative)

    filesystem_paths = set()
    pending = [(source, Path())]
    while pending:
        directory, relative_directory = pending.pop()
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: os.fsencode(entry.name))
        except OSError as exc:
            raise StageError(f"cannot enumerate source directory {directory}: {exc}") from exc
        for entry in entries:
            relative = relative_directory / entry.name
            if entry.name == ".git":
                continue
            try:
                is_directory = entry.is_dir(follow_symlinks=False)
            except OSError as exc:
                raise StageError(f"cannot inspect source entry {entry.path}: {exc}") from exc
            if is_directory:
                excluded_at_root = len(relative.parts) == 1 and entry.name in DEFAULT_EXCLUDED_ROOT_DIRECTORIES
                excluded_by_name = entry.name in DEFAULT_EXCLUDED_DIRECTORY_NAMES
                if excluded_at_root or excluded_by_name:
                    continue
                pending.append((Path(entry.path), relative))
            else:
                # Symlinks, regular files, and special entries are included.
                # The manifest later rejects special entries rather than
                # silently omitting an input it cannot reproduce.
                filesystem_paths.add(relative)

    paths = tracked_paths | filesystem_paths
    return tuple(sorted(paths, key=lambda path: os.fsencode(path.as_posix())))


def _entry_identity(path: Path) -> dict[str, object]:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return {"kind": "missing"}
    if stat.S_ISLNK(mode):
        return {"kind": "symlink", "target": os.readlink(path)}
    if not stat.S_ISREG(mode):
        raise StageError(f"unsupported special source entry: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"kind": "file", "sha256": digest.hexdigest(), "executable": bool(mode & stat.S_IXUSR)}


def _manifest(source: Path, paths: tuple[Path, ...]) -> dict[str, dict[str, object]]:
    return {path.as_posix(): _entry_identity(source / path) for path in paths}


def _clear_worktree(destination: Path) -> None:
    for entry in destination.iterdir():
        if entry.name == ".git":
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def _copy_worktree(source: Path, destination: Path, paths: tuple[Path, ...]) -> None:
    _clear_worktree(destination)
    for relative in paths:
        source_path = source / relative
        try:
            mode = source_path.lstat().st_mode
        except FileNotFoundError:
            continue
        destination_path = destination / relative
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if stat.S_ISLNK(mode):
            destination_path.symlink_to(os.readlink(source_path))
        elif stat.S_ISREG(mode):
            shutil.copy2(source_path, destination_path, follow_symlinks=False)
        else:
            raise StageError(f"unsupported special source entry: {source_path}")


def _verify_stage(destination: Path, source_common_dir: Path, head: str,
                  expected_manifest: dict[str, dict[str, object]]) -> None:
    if not (destination / ".git").is_dir():
        raise StageError("staged .git must be an ordinary directory, not a linked-worktree pointer")
    staged_git_dir = _absolute_git_path(destination, _git(destination, "rev-parse", "--git-dir").strip())
    staged_common_dir = _absolute_git_path(destination, _git(destination, "rev-parse", "--git-common-dir").strip())
    if staged_git_dir != destination / ".git" or staged_common_dir != destination / ".git":
        raise StageError("staged Git metadata is not self-contained under the staged repository")
    if staged_common_dir == source_common_dir:
        raise StageError("staged Git metadata still depends on the source repository")
    alternates = destination / ".git" / "objects" / "info" / "alternates"
    if alternates.exists() and alternates.read_text(encoding="utf-8").strip():
        raise StageError("staged object database uses an external alternate")
    if _git(destination, "rev-parse", "--verify", "HEAD^{commit}").strip() != head:
        raise StageError("staged HEAD does not equal source HEAD")
    _git(destination, "cat-file", "-e", f"{head}^{{commit}}")
    _git(destination, "fsck", "--connectivity-only", "--no-dangling")
    observed = {relative: _entry_identity(destination / relative) for relative in expected_manifest}
    if observed != expected_manifest:
        raise StageError("staged tracked/untracked working-tree content does not match the source")
    _git(destination, "status", "--porcelain=v1", "--untracked-files=all")


def verify_existing_stage(destination: str | os.PathLike[str], *, expected_head: str = "",
                          expected_manifest_sha256: str = "") -> dict[str, object]:
    """Verify a previously created stage without consulting its source checkout."""

    destination_path = Path(destination).resolve(strict=True)
    if not destination_path.is_dir():
        raise StageError("stage must be a repository directory")
    metadata_path = destination_path / STAGE_METADATA_NAME
    manifest_path = destination_path / STAGE_MANIFEST_NAME
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StageError(f"cannot read staged reproducibility evidence: {exc}") from exc
    if metadata.get("schema_version") != "phase-a-repository-stage-v2":
        raise StageError("staged metadata has an unsupported schema")
    if not isinstance(manifest, dict) or not all(isinstance(path, str) for path in manifest):
        raise StageError("staged working-tree manifest is malformed")
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha256 != metadata.get("working_tree_manifest_sha256"):
        raise StageError("staged manifest does not match its metadata digest")
    if expected_manifest_sha256 and manifest_sha256 != expected_manifest_sha256:
        raise StageError("staged manifest does not match the expected digest")
    head = _git(destination_path, "rev-parse", "--verify", "HEAD^{commit}").strip()
    if head != metadata.get("source_head"):
        raise StageError("staged HEAD does not match its metadata")
    if expected_head and head != expected_head:
        raise StageError("staged HEAD does not match the expected commit")
    if not (destination_path / ".git").is_dir():
        raise StageError("staged .git must be an ordinary directory")
    git_dir = _absolute_git_path(destination_path, _git(destination_path, "rev-parse", "--git-dir").strip())
    common_dir = _absolute_git_path(destination_path,
                                    _git(destination_path, "rev-parse", "--git-common-dir").strip())
    if git_dir != destination_path / ".git" or common_dir != destination_path / ".git":
        raise StageError("staged Git metadata is not self-contained")
    alternates = destination_path / ".git" / "objects" / "info" / "alternates"
    if alternates.exists() and alternates.read_text(encoding="utf-8").strip():
        raise StageError("staged object database uses an external alternate")
    _git(destination_path, "cat-file", "-e", f"{head}^{{commit}}")
    _git(destination_path, "fsck", "--connectivity-only", "--no-dangling")
    expected_paths = {Path(path) for path in manifest}
    observed_paths = set(_source_paths(destination_path))
    generated_paths = {Path(STAGE_METADATA_NAME), Path(STAGE_MANIFEST_NAME)}
    if observed_paths != expected_paths | generated_paths:
        raise StageError("staged working-tree path set drifted from its manifest")
    observed = {path.as_posix(): _entry_identity(destination_path / path) for path in expected_paths}
    if observed != manifest:
        raise StageError("staged working-tree content drifted from its manifest")
    return metadata


def stage_repository(source: str | os.PathLike[str], destination: str | os.PathLike[str],
                     *, command: Sequence[str] = ()) -> tuple[Path, int]:
    """Create and verify an independent stage, then optionally run ``command``."""

    source_path, head, source_common_dir = _validate_source(Path(source))
    destination_path = Path(destination).resolve()
    if destination_path.exists():
        raise StageError("destination already exists; refusing to merge or overwrite a stage")
    try:
        if os.path.commonpath((source_path, destination_path)) == str(source_path):
            raise StageError("destination must be outside the source repository")
    except ValueError as exc:
        raise StageError("source and destination must be on comparable absolute paths") from exc

    paths = _source_paths(source_path)
    before = _manifest(source_path, paths)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        _run(("git", "clone", "--no-local", "--no-checkout", "--quiet", str(source_path), str(destination_path)),
             cwd=destination_path.parent)
        created = True
        if _run(("git", "cat-file", "-e", f"{head}^{{commit}}"), cwd=destination_path, check=False).returncode:
            _run(("git", "fetch", "--quiet", "--no-tags", "origin", head), cwd=destination_path)
        _git(destination_path, "checkout", "--detach", "--force", head)
        _copy_worktree(source_path, destination_path, paths)
        after_paths = _source_paths(source_path)
        after = _manifest(source_path, after_paths)
        if _git(source_path, "rev-parse", "--verify", "HEAD^{commit}").strip() != head:
            raise StageError("source HEAD changed while staging")
        if after_paths != paths or after != before:
            raise StageError("source working-tree inputs changed while staging")
        _verify_stage(destination_path, source_common_dir, head, before)
        manifest_path = destination_path / STAGE_MANIFEST_NAME
        manifest_path.write_text(json.dumps(before, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        metadata = {
            "schema_version": "phase-a-repository-stage-v2",
            "source_head": head,
            "source_git_common_dir": str(source_common_dir),
            "staged_git_common_dir": str(destination_path / ".git"),
            "tracked_and_untracked_entries": len(before),
            "default_exclusions": {
                "root_directories": list(DEFAULT_EXCLUDED_ROOT_DIRECTORIES),
                "directory_names_at_any_depth": list(DEFAULT_EXCLUDED_DIRECTORY_NAMES),
            },
            "working_tree_manifest_sha256": hashlib.sha256(
                json.dumps(before, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "command": list(command),
        }
        (destination_path / STAGE_METADATA_NAME).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _verify_stage(destination_path, source_common_dir, head, before)
        verify_existing_stage(destination_path, expected_head=head,
                              expected_manifest_sha256=metadata["working_tree_manifest_sha256"])
        status = 0
        if command:
            status = subprocess.run(tuple(command), cwd=destination_path, check=False).returncode
        return destination_path, status
    except Exception:
        if created and destination_path.exists():
            shutil.rmtree(destination_path)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a verified independent Phase A repository stage")
    parser.add_argument("--source", default=".", help="exact source repository top level")
    parser.add_argument("--destination", help="new stage path outside the source repository")
    parser.add_argument("--verify-existing", help="verify an existing independent stage and exit")
    parser.add_argument("--expected-head", default="", help="required full HEAD for --verify-existing")
    parser.add_argument("--expected-manifest-sha256", default="",
                        help="required working-tree manifest digest for --verify-existing")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="optional bounded or future cleared command, preceded by --")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.verify_existing:
        if args.destination or args.command:
            print("REFUSED: --verify-existing cannot create a stage or run a command", file=sys.stderr)
            return 2
        try:
            metadata = verify_existing_stage(
                args.verify_existing,
                expected_head=args.expected_head,
                expected_manifest_sha256=args.expected_manifest_sha256,
            )
        except (OSError, StageError) as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(metadata, sort_keys=True))
        return 0
    if not args.destination:
        print("REFUSED: --destination is required when creating a stage", file=sys.stderr)
        return 2
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        destination, status = stage_repository(args.source, args.destination, command=command)
    except (OSError, StageError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(destination)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
