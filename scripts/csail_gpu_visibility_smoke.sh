#!/usr/bin/env bash
set -uo pipefail

# Driver-only CSAIL visibility smoke. Deliberately avoids Python, Torch,
# Triton, model code, benchmarks, and scientific/performance payloads.

umask 077

stage_path=${SMOKE_STAGE_PATH:?SMOKE_STAGE_PATH is required}
result_dir=${SMOKE_RESULT_DIR:?SMOKE_RESULT_DIR is required}
expected_head=${SMOKE_EXPECTED_HEAD:?SMOKE_EXPECTED_HEAD is required}
expected_metadata_sha=${SMOKE_EXPECTED_METADATA_SHA256:?SMOKE_EXPECTED_METADATA_SHA256 is required}
expected_partition=${SMOKE_EXPECTED_PARTITION:?SMOKE_EXPECTED_PARTITION is required}
expected_account=${SMOKE_EXPECTED_ACCOUNT:?SMOKE_EXPECTED_ACCOUNT is required}
expected_qos=${SMOKE_EXPECTED_QOS:?SMOKE_EXPECTED_QOS is required}
resource_tuple='nodes=1 ntasks=1 cpus-per-task=2 mem=8G time=00:10:00 gres=gpu:1'

log_path=$result_dir/payload.log
manifest_path=$result_dir/payload-result.env
hash_manifest_path=$result_dir/payload-files.sha256
manifest_tmp=$result_dir/.payload-result.$$.tmp
hash_manifest_tmp=$result_dir/.payload-files.$$.tmp

if [ ! -d "$result_dir" ] || [ -L "$result_dir" ]; then
  printf 'REFUSED: result directory is missing or symlinked: %s\n' "$result_dir" >&2
  exit 2
fi
if [ "$PWD" != "$stage_path" ]; then
  printf 'REFUSED: payload cwd %s is not exact stage %s\n' "$PWD" "$stage_path" >&2
  exit 2
fi
if [ -e "$log_path" ] || [ -L "$log_path" ] ||
   [ -e "$manifest_path" ] || [ -L "$manifest_path" ] ||
   [ -e "$hash_manifest_path" ] || [ -L "$hash_manifest_path" ]; then
  printf 'REFUSED: payload result paths already exist\n' >&2
  exit 2
fi

: > "$log_path"
chmod 0600 "$log_path"

record() {
  printf '%s\n' "$*" >> "$log_path"
}

utc_start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
host=$(hostname)
stage_head=$(git rev-parse --verify HEAD 2>/dev/null || true)
stage_tree=$(git rev-parse --verify 'HEAD^{tree}' 2>/dev/null || true)
stage_git_dir=$(git rev-parse --path-format=absolute --git-dir 2>/dev/null || true)
stage_common_dir=$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)
metadata_path=$stage_path/REPRODUCIBILITY_METADATA.json
metadata_sha=$(sha256sum "$metadata_path" 2>/dev/null | awk '{print $1}' || true)
cuda_visible_devices=${CUDA_VISIBLE_DEVICES-}

record "utc_start=$utc_start"
record "stage_path=$stage_path"
record "stage_head=$stage_head"
record "stage_tree=$stage_tree"
record "stage_git_dir=$stage_git_dir"
record "stage_common_dir=$stage_common_dir"
record "reproducibility_metadata=$metadata_path"
record "reproducibility_metadata_sha256=$metadata_sha"
record "hostname=$host"
record "slurm_job_id=${SLURM_JOB_ID-}"
record "requested_partition=$expected_partition"
record "requested_account=$expected_account"
record "requested_qos=$expected_qos"
record "requested_resources=$resource_tuple"
record "slurm_job_partition=${SLURM_JOB_PARTITION-}"
record "slurm_job_account=${SLURM_JOB_ACCOUNT-}"
record "slurm_job_qos=${SLURM_JOB_QOS-}"
record "slurm_job_num_nodes=${SLURM_JOB_NUM_NODES-}"
record "slurm_ntasks=${SLURM_NTASKS-}"
record "slurm_cpus_per_task=${SLURM_CPUS_PER_TASK-}"
record "slurm_mem_per_node=${SLURM_MEM_PER_NODE-}"
record "slurm_job_gpus=${SLURM_JOB_GPUS-}"
record "slurm_gpus_on_node=${SLURM_GPUS_ON_NODE-}"
record "cuda_visible_devices=$cuda_visible_devices"

set +e
nvidia_list=$(nvidia-smi -L 2>&1)
nvidia_list_exit=$?
nvidia_query=$(nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader,nounits 2>&1)
nvidia_query_exit=$?
set -e

visible_device_count=$(printf '%s\n' "$nvidia_list" | awk '/^GPU [0-9]+:/ {count++} END {print count + 0}')
query_device_count=$(printf '%s\n' "$nvidia_query" | awk 'NF {count++} END {print count + 0}')
if [ -n "$cuda_visible_devices" ] && [ "$cuda_visible_devices" != NoDevFiles ]; then
  assigned_device_count=$(printf '%s\n' "$cuda_visible_devices" | awk -F, '{print NF}')
else
  assigned_device_count=0
fi

record "nvidia_smi_L_exit=$nvidia_list_exit"
record 'nvidia_smi_L_begin'
record "$nvidia_list"
record 'nvidia_smi_L_end'
record "nvidia_smi_query_exit=$nvidia_query_exit"
record 'nvidia_smi_query_fields=model,uuid,total_vram_MiB,driver_version'
record 'nvidia_smi_query_begin'
record "$nvidia_query"
record 'nvidia_smi_query_end'
record "assigned_device_count=$assigned_device_count"
record "visible_device_count=$visible_device_count"
record "query_device_count=$query_device_count"

payload_exit=0
[ "$stage_head" = "$expected_head" ] || payload_exit=1
[ "$stage_git_dir" = "$stage_path/.git" ] || payload_exit=1
[ "$stage_common_dir" = "$stage_path/.git" ] || payload_exit=1
[ -f "$metadata_path" ] && [ ! -L "$metadata_path" ] || payload_exit=1
[ "$metadata_sha" = "$expected_metadata_sha" ] || payload_exit=1
[ -n "${SLURM_JOB_ID-}" ] || payload_exit=1
[ "${SLURM_JOB_PARTITION-}" = "$expected_partition" ] || payload_exit=1
[ "${SLURM_JOB_ACCOUNT-}" = "$expected_account" ] || payload_exit=1
[ "$assigned_device_count" -eq 1 ] || payload_exit=1
[ "$nvidia_list_exit" -eq 0 ] || payload_exit=1
[ "$nvidia_query_exit" -eq 0 ] || payload_exit=1
[ "$visible_device_count" -eq 1 ] || payload_exit=1
[ "$query_device_count" -eq 1 ] || payload_exit=1

if [ "$payload_exit" -eq 0 ]; then
  result=PASS
else
  result=FAIL
fi
utc_end=$(date -u +%Y-%m-%dT%H:%M:%SZ)
record "utc_end=$utc_end"
record "result=$result"
record "payload_exit_code=$payload_exit"

{
  printf 'schema=csail-gpu-visibility-smoke-v2\n'
  printf 'utc_start=%s\n' "$utc_start"
  printf 'utc_end=%s\n' "$utc_end"
  printf 'stage_path=%s\n' "$stage_path"
  printf 'stage_head=%s\n' "$stage_head"
  printf 'stage_tree=%s\n' "$stage_tree"
  printf 'stage_git_dir=%s\n' "$stage_git_dir"
  printf 'stage_common_dir=%s\n' "$stage_common_dir"
  printf 'reproducibility_metadata=%s\n' "$metadata_path"
  printf 'reproducibility_metadata_sha256=%s\n' "$metadata_sha"
  printf 'hostname=%s\n' "$host"
  printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID-}"
  printf 'requested_partition=%s\n' "$expected_partition"
  printf 'requested_account=%s\n' "$expected_account"
  printf 'requested_qos=%s\n' "$expected_qos"
  printf 'requested_resources=%s\n' "$resource_tuple"
  printf 'slurm_job_partition=%s\n' "${SLURM_JOB_PARTITION-}"
  printf 'slurm_job_account=%s\n' "${SLURM_JOB_ACCOUNT-}"
  printf 'slurm_job_qos=%s\n' "${SLURM_JOB_QOS-}"
  printf 'cuda_visible_devices=%s\n' "$cuda_visible_devices"
  printf 'assigned_device_count=%s\n' "$assigned_device_count"
  printf 'nvidia_smi_L_exit=%s\n' "$nvidia_list_exit"
  printf 'nvidia_smi_query_exit=%s\n' "$nvidia_query_exit"
  printf 'visible_device_count=%s\n' "$visible_device_count"
  printf 'query_device_count=%s\n' "$query_device_count"
  printf 'payload_log=%s\n' "$log_path"
  printf 'result=%s\n' "$result"
  printf 'payload_exit_code=%s\n' "$payload_exit"
} > "$manifest_tmp"
chmod 0600 "$manifest_tmp"
mv -- "$manifest_tmp" "$manifest_path"

sha256sum "$log_path" "$manifest_path" > "$hash_manifest_tmp"
chmod 0600 "$hash_manifest_tmp"
mv -- "$hash_manifest_tmp" "$hash_manifest_path"

exit "$payload_exit"
