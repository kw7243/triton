#!/usr/bin/env python3
"""Create or verify the complete non-Git file manifest for an A2 stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Iterable


SCHEMA = "structured-rotations-v2-stage-manifest-v1"
MANIFEST_NAME = "STAGE_FILE_MANIFEST.json"


class ManifestError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _root(value: Path | None) -> Path:
    if value is not None:
        root = value.resolve(strict=True)
    else:
        root = Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"], text=True
            ).strip()
        ).resolve(strict=True)
    if not (root / ".git").is_dir():
        raise ManifestError("stage must have an ordinary .git directory")
    return root


def _paths(root: Path) -> Iterable[Path]:
    for directory, names, files in os.walk(root, topdown=True, followlinks=False):
        current = Path(directory)
        names[:] = sorted(name for name in names if name != ".git")
        for name in names:
            path = current / name
            if path.is_symlink():
                raise ManifestError(f"stage directory symlink is forbidden: {path}")
        for name in sorted(files):
            path = current / name
            relative = path.relative_to(root)
            if relative.as_posix() == MANIFEST_NAME:
                continue
            status = path.lstat()
            if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
                raise ManifestError(f"stage input must be a regular file: {relative}")
            yield path


def collect(root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in _paths(root)
    ]


def write(root: Path) -> Path:
    manifest = root / MANIFEST_NAME
    payload = {
        "schema_version": SCHEMA,
        "root": str(root),
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "tree": subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=root, text=True
        ).strip(),
        "entries": collect(root),
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(manifest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return manifest


def verify(root: Path) -> dict[str, object]:
    manifest = root / MANIFEST_NAME
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA or Path(payload["root"]) != root:
        raise ManifestError("manifest identity differs")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=root, text=True
    ).strip()
    if payload.get("head") != head or payload.get("tree") != tree:
        raise ManifestError("manifest Git identity differs")
    expected = payload.get("entries")
    observed = collect(root)
    if expected != observed:
        raise ManifestError("stage bytes differ from the frozen manifest")
    return {
        "path": str(manifest),
        "sha256": sha256_file(manifest),
        "entries": len(observed),
        "head": head,
        "tree": tree,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    root = _root(args.root)
    if args.write:
        print(write(root))
    else:
        print(json.dumps(verify(root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
