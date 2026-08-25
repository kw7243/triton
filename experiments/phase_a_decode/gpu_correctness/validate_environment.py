#!/usr/bin/env python3
"""Fail-closed allocated-node validation for Phase A GPU correctness."""

import argparse
import glob
import hashlib
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_SOURCE = Path(
    "/data/scratch-fast/kwen1/compute-native-vq/worktrees/vq-phase-a-gpu-correctness-r1")
EXPECTED_ROOT = Path("/data/scratch-fast/kwen1/compute-native-vq")
EXPECTED_ACCOUNT = "vision-torralba-urops-meng"
EXPECTED_QOS = "vision-torralba-interactive"
EXPECTED_PARTITION = "vision-torralba-rtx3090"


def command(args):
    result = subprocess.run(args, check=True, text=True, capture_output=True, timeout=20)
    return result.stdout.strip()


def storage(path):
    usage = shutil.disk_usage(path)
    return {"path": str(path), "total_bytes": usage.total, "free_bytes": usage.free,
            "readable": os.access(path, os.R_OK), "writable": os.access(path, os.W_OK)}


def git(path, *args):
    return command(["git", "-C", str(path), *args])


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_scontrol(text):
    return dict(token.split("=", 1) for token in text.split() if "=" in token)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"status": "failed", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "checks": {}}
    exit_code = 1
    try:
        source, stage, result = (path.resolve(strict=True)
                                 for path in (args.source, args.stage, args.result))
        checks = report["checks"]
        checks["source_exact"] = source == EXPECTED_SOURCE
        checks["stage_parent"] = stage.parent == EXPECTED_ROOT / "staging"
        checks["result_parent"] = result.parent == EXPECTED_ROOT / "results"
        checks["distinct_paths"] = len({source, stage, result}) == 3
        checks["stage_git_present"] = (stage / ".git").exists()
        checks["stage_top_exact"] = Path(git(stage, "rev-parse", "--show-toplevel")) == stage

        metadata_path = stage / "REPRODUCIBILITY_METADATA.json"
        metadata = json.loads(metadata_path.read_text())
        report["reproducibility"] = {
            "path": str(metadata_path), "sha256": sha256(metadata_path), "metadata": metadata,
            "stage_head": git(stage, "rev-parse", "HEAD"),
            "stage_tree": git(stage, "rev-parse", "HEAD^{tree}"),
            "stage_status": git(stage, "status", "--short"),
        }
        checks["metadata_source"] = metadata.get("source_repo") == str(source)
        checks["metadata_stage"] = metadata.get("staged_repo") == str(stage)
        checks["metadata_commit"] = metadata.get("git_commit_full") == report["reproducibility"]["stage_head"]

        job_id = os.environ.get("SLURM_JOB_ID", "")
        allocation_text = command(["scontrol", "show", "job", "-o", job_id])
        allocation = parse_scontrol(allocation_text)
        report["allocation"] = {"job_id": job_id, "hostname": platform.node(),
                                "scontrol": allocation_text,
                                "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
                                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
        checks["job_id_numeric"] = job_id.isdigit()
        checks["compute_host"] = not platform.node().startswith("slurm-login")
        checks["account"] = allocation.get("Account") == EXPECTED_ACCOUNT
        checks["qos"] = allocation.get("QOS") == EXPECTED_QOS
        checks["partition"] = allocation.get("Partition") == EXPECTED_PARTITION
        checks["one_node"] = allocation.get("NumNodes") == "1"
        checks["one_task"] = allocation.get("NumTasks") == "1"
        checks["four_cpus"] = allocation.get("NumCPUs") == "4"
        checks["sixteen_gib"] = allocation.get("MinMemoryNode") in {"16G", "16384M"}
        checks["fifteen_minutes"] = allocation.get("TimeLimit") == "00:15:00"
        checks["one_gpu"] = "gres/gpu=1" in allocation.get("AllocTRES", "")

        devices = sorted(glob.glob("/dev/nvidia*"))
        query = command(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,"
                         "memory.total,memory.free,temperature.gpu,pstate,power.draw,power.limit",
                         "--format=csv,noheader,nounits"])
        report["nvidia"] = {"device_nodes": devices, "identity_health_query": query,
                            "list": command(["nvidia-smi", "-L"])}
        checks["nvidia_devices"] = bool(devices)
        checks["one_visible_nvidia_smi_gpu"] = len(query.splitlines()) == 1

        meminfo = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(("MemTotal:", "MemAvailable:", "SwapTotal:", "SwapFree:")):
                key, value = line.split(":", 1)
                meminfo[key] = value.strip()
        report["host_memory"] = meminfo
        report["storage"] = {name: storage(path) for name, path in
                             (("source", source), ("stage", stage), ("result", result))}
        checks["storage_access"] = all(item["readable"] and item["writable"]
                                       for item in report["storage"].values())

        import torch
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count()
        torch_data = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                      "cuda_available": cuda_available, "device_count": device_count}
        if cuda_available and device_count == 1:
            properties = torch.cuda.get_device_properties(0)
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            torch_data.update({"device_name": torch.cuda.get_device_name(0),
                               "compute_capability": torch.cuda.get_device_capability(0),
                               "vram_property_total_bytes": properties.total_memory,
                               "vram_free_bytes": free_bytes, "vram_total_bytes": total_bytes,
                               "bf16_supported": torch.cuda.is_bf16_supported()})
        report["torch_cuda"] = torch_data
        checks["torch_cuda_available"] = cuda_available
        checks["one_torch_cuda_device"] = device_count == 1

        failures = sorted(name for name, passed in checks.items() if not passed)
        if failures:
            raise RuntimeError("failed validation checks: " + ", ".join(failures))
        report["status"] = "passed"
        exit_code = 0
    except BaseException as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        temporary.replace(args.output)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
