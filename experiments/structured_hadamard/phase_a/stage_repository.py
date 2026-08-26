"""Fail-closed full-repository staging for future experimental commands.

The staged checkout has an ordinary, self-contained ``.git`` directory even
when the source is a linked worktree. The source ``HEAD`` is cloned without
local hardlinks, then the complete tracked and non-ignored untracked working
tree is copied and verified before an optional command may run.
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
    output = _git(source, "ls-files", "-z", "--cached", "--others", "--exclude-standard", text=False)
    paths = []
    for raw in output.split(b"\0"):
        if not raw:
            continue
        relative = Path(os.fsdecode(raw))
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] == ".git":
            raise StageError(f"unsafe repository path: {relative}")
        paths.append(relative)
    if len(paths) != len(set(paths)):
        raise StageError("Git returned duplicate tracked/untracked paths")
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
        metadata = {
            "schema_version": "phase-a-repository-stage-v1",
            "source_head": head,
            "source_git_common_dir": str(source_common_dir),
            "staged_git_common_dir": str(destination_path / ".git"),
            "tracked_and_untracked_entries": len(before),
            "working_tree_manifest_sha256": hashlib.sha256(
                json.dumps(before, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "command": list(command),
        }
        (destination_path / "REPRODUCIBILITY_METADATA.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _verify_stage(destination_path, source_common_dir, head, before)
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
    parser.add_argument("--destination", required=True, help="new stage path outside the source repository")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="optional bounded or future cleared command, preceded by --")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
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
