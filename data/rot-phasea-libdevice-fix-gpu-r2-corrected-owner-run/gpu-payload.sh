#!/usr/bin/env bash
set -euo pipefail
umask 077

attempt=/data/scratch-fast/kwen1/structured-hadamard/rot-phasea-libdevice-fix-gpu-r2-corrected-owner-run/attempt-20260901T214700Z-48a972220979
stage=$attempt/stage/20260901T214700Z-48a972220979-code
run_state=$attempt/run-state
clearance=$run_state/clearance.json
python_bin=/data/scratch-fast/kwen1/micromamba/root/envs/causal_forcing/bin/python
output=$attempt/outputs/phase-a-48a972220979
export PYTHONDONTWRITEBYTECODE=1
export TRITON_CACHE_DIR=$run_state/triton-cache
export CUDA_CACHE_PATH=$run_state/cuda-cache
install -d -m 700 "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
exec > >(tee "$attempt/logs/gpu-execution.log") 2>&1

printf 'gpu_payload_started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'hostname=%s\n' "$(hostname -f)"
printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:?missing SLURM_JOB_ID}"
printf 'slurm_job_partition=%s\n' "${SLURM_JOB_PARTITION:-UNSET}"
printf 'slurm_job_account=%s\n' "${SLURM_JOB_ACCOUNT:-UNSET}"
printf 'slurm_job_qos=%s\n' "${SLURM_JOB_QOS:-UNSET}"
printf 'slurm_ntasks=%s\n' "${SLURM_NTASKS:-UNSET}"
printf 'slurm_cpus_per_task=%s\n' "${SLURM_CPUS_PER_TASK:-UNSET}"
printf 'slurm_mem_per_node=%s\n' "${SLURM_MEM_PER_NODE:-UNSET}"
printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-UNSET}"
/usr/bin/scontrol show job -o "$SLURM_JOB_ID"
printf 'nvidia_smi_list_begin\n'
nvidia-smi -L
printf 'nvidia_smi_list_end\n'
printf 'nvidia_smi_identity_begin\n'
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version,compute_cap --format=csv,noheader,nounits
printf 'nvidia_smi_identity_end\n'
printf 'python_path=%s\n' "$python_bin"
printf 'python_realpath=%s\n' "$(readlink -f "$python_bin")"
printf 'python_sha256=%s\n' "$(sha256sum "$(readlink -f "$python_bin")" | awk '{print $1}')"
printf 'stage_head=%s\n' "$(git -C "$stage" rev-parse 'HEAD^{commit}')"

cd "$run_state"
sha256sum -c helper-sha256sums.txt
stat -c '%a %u %U %G %s %n' \
  stage_audit.py ledger.py gpu_preflight.py gpu-payload.sh srun-owner.sh allocation-owner.sh remote-attempt.sh
sha256sum \
  "$stage/experiments/structured_hadamard/phase_a/activation_quantizer.py" \
  "$stage/experiments/structured_hadamard/phase_a/execute.py" \
  "$stage/experiments/structured_hadamard/phase_a/stage_repository.py" \
  "$stage/REPRODUCIBILITY_MANIFEST.json" \
  "$stage/REPRODUCIBILITY_METADATA.json" \
  "$clearance"
test ! -e "$output"

"$python_bin" - "$run_state/environment.json" <<'PY'
from __future__ import annotations
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

path = Path(sys.argv[1])
if path.exists():
    raise RuntimeError(f"environment record already exists: {path}")
sensitive = ("TOKEN", "SECRET", "PASS", "CREDENTIAL", "COOKIE", "AUTH", "PRIVATE_KEY")
environment = {
    key: ("<redacted>" if any(marker in key.upper() for marker in sensitive) else value)
    for key, value in sorted(os.environ.items())
}
payload = {
    "schema_version": "rot-phasea-corrected-environment-v1",
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "environment": environment,
}
encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "wb") as stream:
    stream.write(encoded)
    stream.flush()
    os.fsync(stream.fileno())
PY

cd "$stage"
"$python_bin" "$run_state/gpu_preflight.py" "$stage" "$clearance" "$run_state/hardware-preflight.json"
test ! -e "$output"
"$run_state/ledger.py" "$run_state/ledgers.json" driver_invocations
printf 'accepted_driver_invocation_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$python_bin" -m experiments.structured_hadamard.phase_a.execute \
  --scheduler-clearance-file "$clearance"
printf 'accepted_driver_finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
sha256sum "$output/phase-a.jsonl" "$output/phase-a.raw-samples.jsonl" "$output/phase-a.execution.json"
printf 'gpu_payload_finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
