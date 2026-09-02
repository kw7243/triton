#!/usr/bin/env python3
"""Fail-closed allocated-node validation bound to the immutable launch manifest."""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from prepare_clean_rerun import (
    DEFAULT_APPROVED_ROOT,
    inventory_digest,
    recursive_inventory,
    sha256,
    verify_self_contained_repo,
)
from result_protocol import atomic_write_json


def command(args: list[str]) -> str:
    result = subprocess.run(args, check=True, text=True, capture_output=True, timeout=20)
    return result.stdout.strip()


def storage(path: Path) -> dict[str, object]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_bytes": usage.total,
        "free_bytes": usage.free,
        "readable": os.access(path, os.R_OK),
        "writable": os.access(path, os.W_OK),
    }


def parse_scontrol(text: str) -> dict[str, str]:
    return dict(token.split("=", 1) for token in text.split() if "=" in token)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report: dict[str, object] = {
        "schema": "vq-phase-a-environment-validation/v2",
        "status": "failed",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "failure_phase": "payload_before_cuda",
        "launch_manifest_sha256": args.manifest_sha256,
        "checks": {},
    }
    exit_code = 1
    try:
        root = DEFAULT_APPROVED_ROOT.resolve(strict=True)
        source, stage, result, manifest_path = (
            path.resolve(strict=True)
            for path in (args.source, args.stage, args.result, args.manifest)
        )
        checks = report["checks"]
        assert isinstance(checks, dict)
        checks["source_parent"] = source.parent == root / "standalone-sources"
        checks["stage_parent"] = stage.parent == root / "staging"
        checks["result_parent"] = result.parent == root / "results"
        checks["distinct_paths"] = len({source, stage, result}) == 3
        checks["manifest_in_result"] = manifest_path == result / "launch_manifest.json"
        checks["manifest_digest"] = sha256(manifest_path) == args.manifest_sha256
        checks["manifest_environment"] = (
            os.environ.get("PHASE_A_MANIFEST_SHA256") == args.manifest_sha256
        )

        manifest = json.loads(manifest_path.read_text())
        report["launch_manifest"] = {
            "path": str(manifest_path),
            "sha256": sha256(manifest_path),
            "commit": manifest.get("commit"),
            "tree": manifest.get("tree"),
        }
        checks["manifest_paths"] = (
            manifest.get("source_repo") == str(source)
            and manifest.get("staged_repo") == str(stage)
            and manifest.get("result_root") == str(result)
        )
        checks["python_identity"] = (
            Path(sys.executable).resolve()
            == Path(str(manifest["benchmark_argv"][0])).resolve(strict=True)
            == Path(os.environ.get("PHASE_A_PYTHON", "")).resolve(strict=True)
        )
        stage_identity = verify_self_contained_repo(
            stage,
            str(manifest["commit"]),
            allowed_untracked=("REPRODUCIBILITY_METADATA.json",),
            compare_objects_with=source,
        )
        report["stage_identity"] = stage_identity
        checks["stage_tree"] = stage_identity["tree"] == manifest.get("tree")
        checks["stage_inventory"] = (
            inventory_digest(recursive_inventory(stage))
            == manifest.get("stage_inventory_sha256")
        )

        scheduler = manifest["scheduler_selection"]
        resources = scheduler["resources"]
        job_id = os.environ.get("SLURM_JOB_ID", "")
        allocation_text = command(["scontrol", "show", "job", "-o", job_id])
        allocation = parse_scontrol(allocation_text)
        report["allocation"] = {
            "job_id": job_id,
            "hostname": platform.node(),
            "scontrol": allocation_text,
            "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        checks["job_id_numeric"] = job_id.isdigit()
        checks["compute_host"] = not platform.node().startswith("slurm-login")
        checks["account"] = allocation.get("Account") == scheduler["account"]
        checks["qos"] = allocation.get("QOS") == scheduler["qos"]
        checks["partition"] = allocation.get("Partition") == scheduler["partition"]
        checks["one_node"] = allocation.get("NumNodes") == str(resources["nodes"])
        checks["one_task"] = allocation.get("NumTasks") == str(resources["tasks"])
        checks["cpus"] = allocation.get("NumCPUs") == str(resources["cpus"])
        checks["memory"] = allocation.get("MinMemoryNode") in {
            f"{resources['memory_gib']}G",
            f"{resources['memory_gib'] * 1024}M",
        }
        checks["time_limit"] = allocation.get("TimeLimit") == f"00:{resources['time_minutes']:02d}:00"
        checks["one_gpu_allocated"] = "gres/gpu=1" in allocation.get("AllocTRES", "")
        checks["requeue_disabled"] = allocation.get("Requeue") == "0"

        devices = sorted(glob.glob("/dev/nvidia*"))
        query = command(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,memory.total,memory.free,"
                "temperature.gpu,pstate,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ]
        )
        listed = command(["nvidia-smi", "-L"])
        report["nvidia"] = {
            "device_nodes": devices,
            "identity_health_query": query,
            "list": listed,
        }
        checks["nvidia_devices"] = bool(devices)
        checks["one_visible_nvidia_smi_gpu"] = len(query.splitlines()) == 1

        meminfo = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(("MemTotal:", "MemAvailable:", "SwapTotal:", "SwapFree:")):
                key, value = line.split(":", 1)
                meminfo[key] = value.strip()
        report["host_memory"] = meminfo
        report["storage"] = {
            name: storage(path)
            for name, path in (("source", source), ("stage", stage), ("result", result))
        }
        checks["storage_access"] = (
            report["storage"]["source"]["readable"]
            and report["storage"]["stage"]["readable"]
            and report["storage"]["result"]["readable"]
            and report["storage"]["result"]["writable"]
        )

        import torch
        import triton

        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count()
        torch_data: dict[str, object] = {
            "torch": torch.__version__,
            "triton": triton.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": cuda_available,
            "device_count": device_count,
        }
        if cuda_available and device_count == 1:
            properties = torch.cuda.get_device_properties(0)
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            torch_data.update(
                device_name=torch.cuda.get_device_name(0),
                compute_capability=torch.cuda.get_device_capability(0),
                vram_property_total_bytes=properties.total_memory,
                vram_free_bytes=free_bytes,
                vram_total_bytes=total_bytes,
                bf16_supported=torch.cuda.is_bf16_supported(),
            )
        report["torch_cuda"] = torch_data
        adequacy = scheduler["adequacy"]
        checks["torch_cuda_available"] = cuda_available
        checks["one_torch_cuda_device"] = device_count == 1
        checks["advertised_gpu_match"] = (
            torch_data.get("device_name") in adequacy["advertised_gpu_names"]
        )
        checks["minimum_vram"] = (
            int(torch_data.get("vram_total_bytes", 0)) >= adequacy["minimum_vram_bytes"]
        )

        failures = sorted(name for name, passed in checks.items() if not passed)
        if failures:
            raise RuntimeError("failed validation checks: " + ", ".join(failures))
        report["status"] = "passed"
        exit_code = 0
    except BaseException as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        atomic_write_json(args.output, report)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
