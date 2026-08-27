#!/usr/bin/env bash

# Shared constants and fail-closed checks for the three submission-origin lanes.
# This file is sourced; callers choose whether to submit, observe, or validate.

ORIGIN_REMOTE_USER=kwen1
ORIGIN_REMOTE_HOST=slurm-login.csail.mit.edu
ORIGIN_REMOTE_TARGET="${ORIGIN_REMOTE_USER}@${ORIGIN_REMOTE_HOST}"
ORIGIN_FORBIDDEN_EVENT_JOB=1579631
ORIGIN_REQUIRED_BASE=d57acb60db2a4507bbff984fb3c9771e8a6ada3d
ORIGIN_MARKER_TOKEN=origin-control-marker-r1-6c6043979a4a4a32
ORIGIN_JOB_NAME=origin-control-marker-r1

ORIGIN_CONTROL_RELATIVE=experiments/structured_hadamard/phase_a/origin_control
ORIGIN_MARKER_RELATIVE="$ORIGIN_CONTROL_RELATIVE/run-marker.sh"
ORIGIN_SUBMIT_RELATIVE="$ORIGIN_CONTROL_RELATIVE/submit-origin-once.sh"
ORIGIN_LIBRARY_RELATIVE="$ORIGIN_CONTROL_RELATIVE/origin-control-lib.sh"
ORIGIN_CONDITION_RELATIVE="$ORIGIN_CONTROL_RELATIVE/terminal-condition.sh"
ORIGIN_ACTION_RELATIVE="$ORIGIN_CONTROL_RELATIVE/terminal-action.sh"
ORIGIN_VALIDATE_RELATIVE="$ORIGIN_CONTROL_RELATIVE/validate-origin-control.sh"
ORIGIN_STAGE_HELPER_RELATIVE=experiments/structured_hadamard/phase_a/stage_repository.py

origin_die() {
  printf 'REFUSED: %s\n' "$1" >&2
  return 2
}

origin_validate_lane() {
  case "${1-}" in
    direct|main-crew|secondmate-crew) return 0 ;;
    *) origin_die "lane must be exactly direct, main-crew, or secondmate-crew" ;;
  esac
}

origin_validate_job_id() {
  local job_id=${1-}
  case "$job_id" in
    ''|*[!0-9]*) origin_die "job id must be numeric" ;;
    "$ORIGIN_FORBIDDEN_EVENT_JOB") origin_die "event-owned job $ORIGIN_FORBIDDEN_EVENT_JOB is outside this experiment" ;;
    *) return 0 ;;
  esac
}

origin_resolve_layout() {
  local library_dir stage_parent experiment_root expected_stage
  library_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P) || return 2
  ORIGIN_STAGE_ROOT=$(git -C "$library_dir" rev-parse --show-toplevel 2>/dev/null) \
    || { origin_die "cannot resolve staged repository root"; return; }
  ORIGIN_STAGE_ROOT=$(cd "$ORIGIN_STAGE_ROOT" && pwd -P) || return 2
  stage_parent=${ORIGIN_STAGE_ROOT%/*}
  experiment_root=${stage_parent%/*}
  expected_stage="$experiment_root/staging/${ORIGIN_STAGE_ROOT##*/}"
  [ "$stage_parent" = "$experiment_root/staging" ] \
    || { origin_die "stage is not inside the experiment staging directory"; return; }
  [ "$ORIGIN_STAGE_ROOT" = "$expected_stage" ] \
    || { origin_die "stage path did not resolve canonically"; return; }
  case "$experiment_root" in
    /data/scratch-fast/kwen1/structured-hadamard/gpu-submit-origin-r1-[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]T[0-9][0-9][0-9][0-9][0-9][0-9]Z) ;;
    *) origin_die "experiment root is outside the fixed timestamped scratch namespace"; return ;;
  esac
  case "${ORIGIN_STAGE_ROOT##*/}" in
    [0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]T[0-9][0-9][0-9][0-9][0-9][0-9]Z-[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]-code) ;;
    *) origin_die "stage basename is not the fixed timestamp-commit form"; return ;;
  esac
  ORIGIN_EXPERIMENT_ROOT=$experiment_root
  ORIGIN_RESULT_ROOT="$experiment_root/results"
  ORIGIN_STAGE_SEAL="$ORIGIN_RESULT_ROOT/stage-seal.env"
  ORIGIN_OUTPUT_TEMPLATE="$ORIGIN_RESULT_ROOT/slurm-%j.out"
  ORIGIN_MARKER="$ORIGIN_STAGE_ROOT/$ORIGIN_MARKER_RELATIVE"
  ORIGIN_SUBMIT="$ORIGIN_STAGE_ROOT/$ORIGIN_SUBMIT_RELATIVE"
  ORIGIN_CONDITION="$ORIGIN_STAGE_ROOT/$ORIGIN_CONDITION_RELATIVE"
  ORIGIN_ACTION="$ORIGIN_STAGE_ROOT/$ORIGIN_ACTION_RELATIVE"
}

origin_mode() { stat -c '%a' -- "$1" 2>/dev/null; }
origin_owner() { stat -c '%U' -- "$1" 2>/dev/null; }
origin_sha256() { sha256sum -- "$1" | awk '{print $1}'; }

origin_seal_value() {
  local key=$1 count value
  count=$(awk -F= -v key="$key" '$1 == key { count++ } END { print count + 0 }' "$ORIGIN_STAGE_SEAL") || return 2
  [ "$count" -eq 1 ] || { origin_die "stage seal key is missing or duplicated: $key"; return; }
  value=$(awk -F= -v key="$key" '$1 == key { sub(/^[^=]*=/, ""); print }' "$ORIGIN_STAGE_SEAL") || return 2
  case "$value" in *$'\n'*) origin_die "stage seal value contains a newline: $key"; return ;; esac
  printf '%s\n' "$value"
}

origin_require_seal_value() {
  local key=$1 expected=$2 observed
  observed=$(origin_seal_value "$key") || return 2
  [ "$observed" = "$expected" ] \
    || { origin_die "stage seal drift for $key"; return; }
}

origin_require_hash() {
  local key=$1 path=$2 expected observed
  expected=$(origin_seal_value "$key") || return 2
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] \
    || { origin_die "stage seal hash is malformed: $key"; return; }
  observed=$(origin_sha256 "$path") || { origin_die "cannot hash $path"; return; }
  [ "$observed" = "$expected" ] || { origin_die "hash drift for $path"; return; }
}

origin_preflight_stage() {
  local head manifest_digest writable
  origin_resolve_layout || return 2
  [ "$(id -un)" = "$ORIGIN_REMOTE_USER" ] || { origin_die "submission user is not $ORIGIN_REMOTE_USER"; return; }
  [ -d "$ORIGIN_STAGE_ROOT" ] && [ ! -L "$ORIGIN_STAGE_ROOT" ] \
    || { origin_die "stage is missing or symlinked"; return; }
  [ -d "$ORIGIN_RESULT_ROOT" ] && [ ! -L "$ORIGIN_RESULT_ROOT" ] \
    || { origin_die "result root is missing or symlinked"; return; }
  [ "$(origin_owner "$ORIGIN_STAGE_ROOT")" = "$ORIGIN_REMOTE_USER" ] \
    || { origin_die "stage owner drifted"; return; }
  [ "$(origin_owner "$ORIGIN_RESULT_ROOT")" = "$ORIGIN_REMOTE_USER" ] \
    || { origin_die "result-root owner drifted"; return; }
  [ "$(origin_mode "$ORIGIN_RESULT_ROOT")" = 700 ] \
    || { origin_die "result root must remain mode 0700"; return; }
  [ -f "$ORIGIN_STAGE_SEAL" ] && [ ! -L "$ORIGIN_STAGE_SEAL" ] \
    || { origin_die "stage seal is missing or symlinked"; return; }
  [ "$(origin_mode "$ORIGIN_STAGE_SEAL")" = 400 ] && [ "$(origin_owner "$ORIGIN_STAGE_SEAL")" = "$ORIGIN_REMOTE_USER" ] \
    || { origin_die "stage seal permissions or owner drifted"; return; }
  origin_require_seal_value schema origin-control-stage-seal-v1 || return 2
  origin_require_seal_value stage "$ORIGIN_STAGE_ROOT" || return 2
  origin_require_seal_value result_root "$ORIGIN_RESULT_ROOT" || return 2
  origin_require_seal_value remote_target "$ORIGIN_REMOTE_TARGET" || return 2
  origin_require_seal_value branch fm/gpu-submit-origin-maincrew-r1 || return 2
  origin_require_seal_value lineage_base "$ORIGIN_REQUIRED_BASE" || return 2
  origin_require_seal_value marker_token "$ORIGIN_MARKER_TOKEN" || return 2
  head=$(origin_seal_value source_head) || return 2
  [[ "$head" =~ ^[0-9a-f]{40}$ ]] || { origin_die "sealed source commit is malformed"; return; }
  [ "${ORIGIN_STAGE_ROOT##*/}" = "${ORIGIN_EXPERIMENT_ROOT##*-}-${head:0:12}-code" ] \
    || { origin_die "stage basename does not bind its timestamp and source commit"; return; }
  git -C "$ORIGIN_STAGE_ROOT" merge-base --is-ancestor "$ORIGIN_REQUIRED_BASE" "$head" \
    || { origin_die "staged commit is outside the required completed marker lineage"; return; }
  origin_require_hash marker_sha256 "$ORIGIN_MARKER" || return 2
  origin_require_hash submit_sha256 "$ORIGIN_SUBMIT" || return 2
  origin_require_hash library_sha256 "$ORIGIN_STAGE_ROOT/$ORIGIN_LIBRARY_RELATIVE" || return 2
  origin_require_hash condition_sha256 "$ORIGIN_CONDITION" || return 2
  origin_require_hash action_sha256 "$ORIGIN_ACTION" || return 2
  origin_require_hash validate_sha256 "$ORIGIN_STAGE_ROOT/$ORIGIN_VALIDATE_RELATIVE" || return 2
  origin_require_hash stage_helper_sha256 "$ORIGIN_STAGE_ROOT/$ORIGIN_STAGE_HELPER_RELATIVE" || return 2
  origin_require_hash metadata_sha256 "$ORIGIN_STAGE_ROOT/REPRODUCIBILITY_METADATA.json" || return 2
  origin_require_hash manifest_file_sha256 "$ORIGIN_STAGE_ROOT/REPRODUCIBILITY_MANIFEST.json" || return 2
  manifest_digest=$(origin_seal_value working_tree_manifest_sha256) || return 2
  PYTHONDONTWRITEBYTECODE=1 python3 "$ORIGIN_STAGE_ROOT/$ORIGIN_STAGE_HELPER_RELATIVE" \
    --verify-existing "$ORIGIN_STAGE_ROOT" --expected-head "$head" \
    --expected-manifest-sha256 "$manifest_digest" >/dev/null \
    || { origin_die "complete staged repository verification failed"; return; }
  writable=$(find "$ORIGIN_STAGE_ROOT" -perm /222 -print -quit) || return 2
  [ -z "$writable" ] || { origin_die "immutable stage contains a writable path: $writable"; return; }
  for path in "$ORIGIN_RESULT_ROOT/latches" "$ORIGIN_RESULT_ROOT/ledgers" \
              "$ORIGIN_RESULT_ROOT/terminal-latches" \
              "$ORIGIN_RESULT_ROOT/terminal-captures"; do
    [ -d "$path" ] && [ ! -L "$path" ] && [ "$(origin_mode "$path")" = 700 ] \
      || { origin_die "result subdirectory is missing, symlinked, or not mode 0700: $path"; return; }
  done
  [ ! -e "$ORIGIN_RESULT_ROOT/%j.out" ] && [ ! -L "$ORIGIN_RESULT_ROOT/%j.out" ] \
    || { origin_die "literal output-template path collides"; return; }
}

origin_build_sbatch_argv() {
  # shellcheck disable=SC2034 # consumed by scripts sourcing this library
  ORIGIN_SBATCH_ARGV=(
    /usr/bin/sbatch
    --parsable
    --account=vision-torralba-urops-meng
    --qos=vision-torralba-interactive
    --partition=vision-torralba-rtx3090
    --nodes=1
    --ntasks=1
    --cpus-per-task=4
    --mem=16G
    --time=01:00:00
    --gres=gpu:1
    "--job-name=$ORIGIN_JOB_NAME"
    "--chdir=$ORIGIN_STAGE_ROOT"
    --export=NONE
    "--output=$ORIGIN_OUTPUT_TEMPLATE"
    "--error=$ORIGIN_OUTPUT_TEMPLATE"
    "$ORIGIN_MARKER"
  )
}

origin_build_terminal_owner_argv() {
  local job_id=$1
  if [ "$job_id" != __JOB_ID__ ]; then
    origin_validate_job_id "$job_id" || return 2
  fi
  # shellcheck disable=SC2034 # consumed by scripts sourcing this library
  ORIGIN_TERMINAL_OWNER_ARGV=(
    /home/ubuntu/firstmate/bin/fm-procevent-when.sh
    arm "slurm-$job_id"
    --interval 30 --stable 2 --deadline 172800
    --condition-timeout 30 --action-timeout 300 --error-budget 3
    --condition /usr/bin/ssh -oBatchMode=yes -oConnectTimeout=15
    "$ORIGIN_REMOTE_TARGET" "$ORIGIN_CONDITION" "$job_id"
    --action /usr/bin/ssh -oBatchMode=yes -oConnectTimeout=15
    "$ORIGIN_REMOTE_TARGET" "$ORIGIN_ACTION" "$job_id"
  )
}

origin_assert_lane_fresh() {
  local lane=$1 path
  origin_validate_lane "$lane" || return 2
  for path in "$ORIGIN_RESULT_ROOT/latches/$lane" \
              "$ORIGIN_RESULT_ROOT/ledgers/$lane.txt"; do
    case "$path" in "$ORIGIN_RESULT_ROOT"/*) ;; *) origin_die "unsafe lane result path"; return ;; esac
    [ ! -e "$path" ] && [ ! -L "$path" ] || { origin_die "lane result path already exists: $path"; return; }
  done
}

origin_claim_lane_once() {
  local lane=$1 latch="$ORIGIN_RESULT_ROOT/latches/$1"
  origin_assert_lane_fresh "$lane" || return 2
  (umask 077; mkdir "$latch") 2>/dev/null \
    || { origin_die "lane latch already exists or cannot be created: $latch"; return; }
  printf '%s\n' "$latch"
}

origin_terminal_state() {
  case "$1" in
    BOOT_FAIL|CANCELLED|COMPLETED|DEADLINE|FAILED|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|REVOKED|SPECIAL_EXIT|TIMEOUT) return 0 ;;
    *) return 1 ;;
  esac
}
