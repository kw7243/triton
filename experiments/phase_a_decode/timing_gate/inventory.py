#!/usr/bin/env python3
"""NUL-safe, content-addressed inventory for a frozen repository stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


SCHEMA = "vq-phase-a-stage-inventory/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def records(root: Path) -> list[dict[str, Any]]:
    root = root.resolve(strict=True)
    output: list[dict[str, Any]] = []
    for directory, names, files in os.walk(root, topdown=True, followlinks=False):
        names.sort()
        files.sort()
        base = Path(directory)
        for name in [*names, *files]:
            path = base / name
            info = path.lstat()
            relative = os.fsdecode(os.fsencode(path.relative_to(root)))
            common = {
                "path": relative,
                "mode": format(stat.S_IMODE(info.st_mode), "04o"),
                "uid": info.st_uid,
                "gid": info.st_gid,
                "bytes": info.st_size,
                "nlink": info.st_nlink,
            }
            if stat.S_ISDIR(info.st_mode):
                common["type"] = "directory"
            elif stat.S_ISREG(info.st_mode):
                if info.st_nlink != 1:
                    raise ValueError(f"shared hard-linked regular file: {relative}")
                common.update(type="file", sha256=_sha256(path))
            elif stat.S_ISLNK(info.st_mode):
                target = os.readlink(path)
                common.update(
                    type="symlink",
                    target=target,
                    sha256=hashlib.sha256(
                        b"symlink\0" + os.fsencode(target)
                    ).hexdigest(),
                )
            else:
                raise ValueError(f"unsupported stage entry type: {relative}")
            output.append(common)
    return output


def content_digest(items: list[dict[str, Any]]) -> str:
    encoded = json.dumps(
        items, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def build(root: Path) -> dict[str, Any]:
    items = records(root)
    return {
        "schema": SCHEMA,
        "root": str(root.resolve(strict=True)),
        "record_count": len(items),
        "content_bytes": sum(
            int(item["bytes"]) for item in items if item["type"] == "file"
        ),
        "content_digest": content_digest(items),
        "records": items,
    }


def atomic_write(path: Path, value: dict[str, Any], mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def file_sha256(path: Path) -> str:
    return _sha256(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expect-digest")
    args = parser.parse_args()
    value = build(args.root)
    if args.expect_digest and value["content_digest"] != args.expect_digest:
        raise RuntimeError(
            f"stage inventory changed: {value['content_digest']} != "
            f"{args.expect_digest}"
        )
    if args.output:
        atomic_write(args.output, value)
    else:
        print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
