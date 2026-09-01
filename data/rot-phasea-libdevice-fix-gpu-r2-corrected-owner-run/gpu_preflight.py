#!/usr/bin/env python3
"""One-device CUDA visibility and untimed corrected-A4 correctness gate."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys


stage = Path(sys.argv[1]).resolve(strict=True)
clearance = Path(sys.argv[2]).resolve(strict=True)
record_path = Path(sys.argv[3])
if record_path.exists():
    raise RuntimeError(f"hardware preflight record already exists: {record_path}")
if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
    raise RuntimeError("CUDA_VISIBLE_DEVICES is empty or unset")
sys.path.insert(0, str(stage))

import torch
import triton
from experiments.structured_hadamard.phase_a import activation_quantizer
from experiments.structured_hadamard.phase_a.activation_quantizer import (
    W4A4ActivationQuantizer,
    quantize_rows_reference,
)
from experiments.structured_hadamard.phase_a.execute import CudaRuntime, _errors, prepare_invocation


if not torch.cuda.is_available():
    raise RuntimeError("Torch CUDA is unavailable")
if torch.cuda.device_count() != 1:
    raise RuntimeError(f"expected exactly one Torch CUDA device, got {torch.cuda.device_count()}")
torch.cuda.set_device(0)
properties = torch.cuda.get_device_properties(0)
probe = torch.ones(1, device="cuda:0") + 1.0
torch.cuda.synchronize()
if probe.item() != 2.0:
    raise RuntimeError("synchronized CUDA tensor operation produced the wrong value")

authorization = prepare_invocation(clearance, root=stage)
runtime = CudaRuntime(authorization.device)
source = runtime.fixed_input()
if tuple(source.shape) != (1, 11008) or source.dtype != torch.float16 or not source.is_cuda:
    raise RuntimeError(f"fixed input contract mismatch: {source.shape} {source.dtype} {source.device}")
source_rows = source.detach().to(dtype=torch.float32, device="cpu").tolist()
quantizer = W4A4ActivationQuantizer(source)
observed = quantizer(source)
torch.cuda.synchronize()
observed_rows = observed.detach().to(dtype=torch.float32, device="cpu").tolist()
expected_rows, expected_scales = quantize_rows_reference(source_rows)
relative_error, max_abs_error = _errors(observed_rows, expected_rows)
if relative_error > 5e-3 or max_abs_error > 5e-2:
    raise RuntimeError(f"fixed A4 correctness failed: relative={relative_error}, max_abs={max_abs_error}")


def file_identity(path: str | os.PathLike[str]) -> dict[str, object]:
    resolved = Path(path).resolve()
    data = resolved.read_bytes()
    return {"path": str(resolved), "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


value = {
    "schema_version": "rot-phasea-corrected-gpu-preflight-v1",
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "slurm": {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "account": os.environ.get("SLURM_JOB_ACCOUNT"),
        "qos": os.environ.get("SLURM_JOB_QOS"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "mem_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
    },
    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    "torch_cuda_device_count": torch.cuda.device_count(),
    "synchronized_cuda_tensor_value": probe.item(),
    "fixed_input": {"shape": list(source.shape), "dtype": str(source.dtype), "seed": 0},
    "gpu": {
        "name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
        "multiprocessor_count": properties.multi_processor_count,
    },
    "versions": {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
    },
    "modules": {
        "python": file_identity(sys.executable),
        "torch": file_identity(torch.__file__),
        "triton": file_identity(triton.__file__),
        "activation_quantizer": file_identity(activation_quantizer.__file__),
    },
    "stage_head": authorization.stage.head,
    "stage_manifest_sha256": authorization.stage.manifest_sha256,
    "clearance_sha256": authorization.clearance_sha256,
    "untimed_a4_correctness": {
        "passed": True,
        "relative_error": relative_error,
        "max_abs_error": max_abs_error,
        "reference_scale": expected_scales[0],
        "rounding": "nearest-even",
    },
}
encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
descriptor = os.open(record_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "wb") as stream:
    stream.write(encoded)
    stream.flush()
    os.fsync(stream.fileno())
print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
print("UNTIMED_A4_CORRECTNESS=PASS")
