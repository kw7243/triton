#!/usr/bin/env python3
"""Fail-closed allocated-node validation for the timing manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "python"))

import torch
import triton

from inventory import build
from runtime import atomic_json


SHARED_FS = ("nfs", "afs", "auristor", "cifs", "smb", "fuse.sshfs")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mount(path: Path) -> dict[str, str]:
    output = subprocess.check_output(
        ["findmnt", "-T", str(path), "-n", "-o", "FSTYPE,TARGET,SOURCE"],
        text=True,
    ).strip()
    fields = output.split(None, 2)
    if len(fields) != 3:
        raise RuntimeError(f"unparseable findmnt output for {path}: {output!r}")
    return {"fstype": fields[0], "target": fields[1], "source": fields[2]}


def pip_freeze() -> str:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.check_output(
        [sys.executable, "-B", "-m", "pip", "freeze"],
        text=True,
        env=environment,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    errors: list[str] = []
    manifest = json.loads(args.manifest.read_text())
    if sha256(args.manifest) != args.manifest_sha256:
        errors.append("launch manifest SHA-256 mismatch")
    paths = manifest["paths"]
    stage = Path(paths["stage"]).resolve(strict=True)
    result = Path(paths["result"]).resolve(strict=True)
    research_root = Path(manifest["storage"]["research_root"]).resolve(strict=True)
    scratch_root = Path(manifest["storage"]["scratch_root"]).resolve(strict=True)
    if not str(result).startswith(str(research_root) + os.sep):
        errors.append("result is outside RESEARCH_ROOT")
    if not str(stage).startswith(str(scratch_root) + os.sep):
        errors.append("stage is outside SCRATCH_ROOT")
    if stage == result:
        errors.append("stage and result are not separated")
    if not (stage / ".git").is_dir():
        errors.append("stage .git is not self-contained")
    for forbidden in (
        stage / ".git" / "commondir",
        stage / ".git" / "objects" / "info" / "alternates",
    ):
        if forbidden.exists():
            errors.append(f"external Git dependency exists: {forbidden}")
    try:
        current_inventory = build(stage)
    except (OSError, ValueError) as exc:
        errors.append(f"stage inventory failed: {exc}")
        current_inventory = {}
    expected_inventory = manifest["stage_inventory"]
    if current_inventory.get("content_digest") != expected_inventory["content_digest"]:
        errors.append("frozen stage inventory changed")
    if current_inventory.get("record_count") != expected_inventory["record_count"]:
        errors.append("frozen stage record count changed")
    if any(
        int(item["mode"], 8) & 0o222
        for item in current_inventory.get("records", [])
    ) or stage.stat().st_mode & 0o222:
        errors.append("frozen stage remains writable")

    selected = manifest["scheduler"]["selected"]
    host = socket.gethostname().split(".")[0]
    if host != selected["node"]:
        errors.append(f"allocated node {host} differs from selected {selected['node']}")
    if os.environ.get("SLURM_JOB_PARTITION") != selected["partition"]:
        errors.append("allocated partition differs from manifest")
    if torch.cuda.device_count() != 1 or not torch.cuda.is_available():
        errors.append("exactly one visible CUDA GPU is required")
    gpu: dict[str, Any] = {}
    if torch.cuda.is_available() and torch.cuda.device_count() == 1:
        properties = torch.cuda.get_device_properties(0)
        capability = list(torch.cuda.get_device_capability(0))
        gpu = {
            "name": properties.name,
            "compute_capability": capability,
            "total_memory_bytes": properties.total_memory,
        }
        if capability != selected["compute_capability"]:
            errors.append("GPU compute capability differs from manifest")
        if not (
            selected["min_device_memory_bytes"]
            <= properties.total_memory
            <= selected["max_device_memory_bytes"]
        ):
            errors.append("GPU memory differs from manifest range")

    identity = manifest["environment"]
    executable = str(Path(sys.executable).resolve(strict=True))
    if executable != identity["python_realpath"]:
        errors.append("Python executable differs from manifest")
    if sha256(Path(sys.executable).resolve(strict=True)) != identity["python_sha256"]:
        errors.append("Python executable bytes differ from manifest")
    observed_versions = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    if observed_versions != identity["versions"]:
        errors.append("Python/PyTorch/Triton/CUDA identity differs from manifest")
    freeze = pip_freeze()
    if hashlib.sha256(freeze.encode()).hexdigest() != identity["pip_freeze_sha256"]:
        errors.append("pip freeze differs from manifest")
    triton_file = str(Path(triton.__file__).resolve(strict=True))
    if not triton_file.startswith(str(stage / "python") + os.sep):
        errors.append("Triton import did not resolve to frozen stage")

    storage_mounts = {
        "research_root": mount(research_root),
        "scratch_root": mount(scratch_root),
    }
    if storage_mounts != manifest["storage"]["mounts"]:
        errors.append("storage mount identity differs from manifest")
    slurm_tmp = os.environ.get("SLURM_TMPDIR", "")
    tmpdir = os.environ.get("TMPDIR", "")
    tmp_mount: dict[str, str] = {}
    if not slurm_tmp or not tmpdir:
        errors.append("SLURM_TMPDIR and TMPDIR are required")
    else:
        slurm_tmp_path = Path(slurm_tmp).resolve(strict=True)
        tmp_path = Path(tmpdir).resolve(strict=True)
        if tmp_path.parent != slurm_tmp_path:
            errors.append("TMPDIR is not the manifest-bound child of SLURM_TMPDIR")
        if tmp_path.name != manifest["tmpdir"]["child_name"]:
            errors.append("TMPDIR child name differs from manifest")
        tmp_mount = mount(tmp_path)
        if tmp_mount["fstype"].lower().startswith(SHARED_FS):
            errors.append(f"TMPDIR is on shared filesystem {tmp_mount['fstype']}")
        if str(tmp_path).startswith(str(research_root)) or str(tmp_path).startswith(str(scratch_root)):
            errors.append("TMPDIR resolves inside shared research storage")

    payload = {
        "schema": "vq-phase-a-timing-environment-validation/v1",
        "status": "passed" if not errors else "failed",
        "launch_manifest_sha256": args.manifest_sha256,
        "host": host,
        "gpu": gpu,
        "versions": observed_versions,
        "triton_file": triton_file,
        "storage_mounts": storage_mounts,
        "tmp_mount": tmp_mount,
        "errors": errors,
    }
    atomic_json(args.output, payload)
    if errors:
        raise RuntimeError("environment validation failed: " + "; ".join(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
