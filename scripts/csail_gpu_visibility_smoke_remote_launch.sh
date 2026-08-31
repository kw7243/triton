#!/usr/bin/env bash
set -euo pipefail

umask 077

attempt=${SMOKE_ATTEMPT_ROOT:?SMOKE_ATTEMPT_ROOT is required}
stage=${SMOKE_STAGE_PATH:?SMOKE_STAGE_PATH is required}
result_dir=${SMOKE_RESULT_DIR:?SMOKE_RESULT_DIR is required}
expected_head=${SMOKE_EXPECTED_HEAD:?SMOKE_EXPECTED_HEAD is required}
expected_metadata_sha=${SMOKE_EXPECTED_METADATA_SHA256:?SMOKE_EXPECTED_METADATA_SHA256 is required}
partition=${SMOKE_EXPECTED_PARTITION:?SMOKE_EXPECTED_PARTITION is required}
account=${SMOKE_EXPECTED_ACCOUNT:?SMOKE_EXPECTED_ACCOUNT is required}
qos=${SMOKE_EXPECTED_QOS:?SMOKE_EXPECTED_QOS is required}

run_state=$attempt/run-state
latch=$run_state/allocation-once.latch
stage_verification=$attempt/preflight/stage-verification.env
salloc_stderr=$result_dir/salloc.stderr.log

[ "$(id -un)" = kwen1 ]
[ -d "$attempt" ] && [ ! -L "$attempt" ]
[ -d "$stage/.git" ] && [ ! -L "$stage/.git" ]
[ -d "$result_dir" ] && [ ! -L "$result_dir" ]
[ -d "$run_state" ] && [ ! -L "$run_state" ]
[ -d "$latch" ] && [ ! -L "$latch" ]
[ -f "$stage_verification" ] && [ ! -L "$stage_verification" ]
grep -Fx 'stage_pass=true' "$stage_verification" >/dev/null
[ "$(git -C "$stage" rev-parse HEAD)" = "$expected_head" ]
[ "$(sha256sum "$stage/REPRODUCIBILITY_METADATA.json" | awk '{print $1}')" = "$expected_metadata_sha" ]
[ ! -e "$salloc_stderr" ] && [ ! -L "$salloc_stderr" ]

salloc_latch=$run_state/salloc-issued.latch
[ ! -e "$salloc_latch" ] && [ ! -L "$salloc_latch" ]
mkdir -m 0700 "$salloc_latch"
printf 'salloc_attempts=1\nissued_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$salloc_latch/attempt.env"
chmod 0400 "$salloc_latch/attempt.env"

set +e
/usr/bin/salloc \
  "--account=$account" \
  "--qos=$qos" \
  "--partition=$partition" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=2 \
  --mem=8G \
  --time=00:10:00 \
  --gres=gpu:1 \
  /usr/bin/env \
  "SMOKE_STAGE_PATH=$stage" \
  "SMOKE_RESULT_DIR=$result_dir" \
  "SMOKE_RUN_STATE=$run_state" \
  "SMOKE_EXPECTED_HEAD=$expected_head" \
  "SMOKE_EXPECTED_METADATA_SHA256=$expected_metadata_sha" \
  "SMOKE_EXPECTED_PARTITION=$partition" \
  "SMOKE_EXPECTED_ACCOUNT=$account" \
  "SMOKE_EXPECTED_QOS=$qos" \
  /bin/bash "$stage/scripts/csail_gpu_visibility_smoke_allocated.sh" \
  2> "$salloc_stderr"
salloc_rc=$?
set -e

chmod 0600 "$salloc_stderr"
terminal_tmp=$run_state/.salloc-terminal.$$.tmp
{
  printf 'schema=csail-gpu-visibility-salloc-terminal-v1\n'
  printf 'terminal_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'salloc_attempts=1\n'
  if [ -d "$run_state/srun-issued.latch" ]; then
    printf 'srun_attempts=1\n'
  else
    printf 'srun_attempts=0\n'
  fi
  printf 'salloc_exit_code=%s\n' "$salloc_rc"
} > "$terminal_tmp"
chmod 0600 "$terminal_tmp"
mv -- "$terminal_tmp" "$run_state/salloc-terminal.env"
exit "$salloc_rc"
