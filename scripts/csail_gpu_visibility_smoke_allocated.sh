#!/usr/bin/env bash
set -euo pipefail

umask 077

stage=${SMOKE_STAGE_PATH:?SMOKE_STAGE_PATH is required}
result_dir=${SMOKE_RESULT_DIR:?SMOKE_RESULT_DIR is required}
run_state=${SMOKE_RUN_STATE:?SMOKE_RUN_STATE is required}
expected_head=${SMOKE_EXPECTED_HEAD:?SMOKE_EXPECTED_HEAD is required}
expected_metadata_sha=${SMOKE_EXPECTED_METADATA_SHA256:?SMOKE_EXPECTED_METADATA_SHA256 is required}
partition=${SMOKE_EXPECTED_PARTITION:?SMOKE_EXPECTED_PARTITION is required}
account=${SMOKE_EXPECTED_ACCOUNT:?SMOKE_EXPECTED_ACCOUNT is required}
qos=${SMOKE_EXPECTED_QOS:?SMOKE_EXPECTED_QOS is required}

[ -d "$stage/.git" ] && [ ! -L "$stage/.git" ]
[ -d "$result_dir" ] && [ ! -L "$result_dir" ]
[ -d "$run_state" ] && [ ! -L "$run_state" ]
[ "$(git -C "$stage" rev-parse HEAD)" = "$expected_head" ]

progress_tmp=$run_state/.allocation-progress.$$.tmp
progress=$run_state/allocation-progress.env
[ ! -e "$progress" ] && [ ! -L "$progress" ]
{
  printf 'schema=csail-gpu-visibility-allocation-progress-v1\n'
  printf 'recorded_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'allocation_granted=true\n'
  printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID-}"
  printf 'partition=%s\n' "${SLURM_JOB_PARTITION-}"
  printf 'account=%s\n' "${SLURM_JOB_ACCOUNT-}"
  printf 'qos=%s\n' "${SLURM_JOB_QOS-}"
  printf 'nodes=%s\n' "${SLURM_JOB_NUM_NODES-}"
  printf 'ntasks=%s\n' "${SLURM_NTASKS-}"
  printf 'cpus_per_task=%s\n' "${SLURM_CPUS_PER_TASK-}"
  printf 'mem_per_node=%s\n' "${SLURM_MEM_PER_NODE-}"
  printf 'job_gpus=%s\n' "${SLURM_JOB_GPUS-}"
} > "$progress_tmp"
chmod 0600 "$progress_tmp"
mv -- "$progress_tmp" "$progress"

srun_latch=$run_state/srun-issued.latch
[ ! -e "$srun_latch" ] && [ ! -L "$srun_latch" ]
mkdir -m 0700 "$srun_latch"
printf 'srun_attempts=1\nissued_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$srun_latch/attempt.env"
chmod 0400 "$srun_latch/attempt.env"

cd "$stage"
set +e
/usr/bin/srun --pty \
  /usr/bin/env \
  "SMOKE_STAGE_PATH=$stage" \
  "SMOKE_RESULT_DIR=$result_dir" \
  "SMOKE_EXPECTED_HEAD=$expected_head" \
  "SMOKE_EXPECTED_METADATA_SHA256=$expected_metadata_sha" \
  "SMOKE_EXPECTED_PARTITION=$partition" \
  "SMOKE_EXPECTED_ACCOUNT=$account" \
  "SMOKE_EXPECTED_QOS=$qos" \
  /bin/bash "$stage/scripts/csail_gpu_visibility_smoke.sh"
srun_rc=$?
set -e

terminal_tmp=$run_state/.srun-terminal.$$.tmp
{
  printf 'schema=csail-gpu-visibility-srun-terminal-v1\n'
  printf 'terminal_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'srun_exit_code=%s\n' "$srun_rc"
} > "$terminal_tmp"
chmod 0600 "$terminal_tmp"
mv -- "$terminal_tmp" "$run_state/srun-terminal.env"
exit "$srun_rc"
