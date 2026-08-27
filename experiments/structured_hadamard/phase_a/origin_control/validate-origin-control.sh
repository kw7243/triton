#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=experiments/structured_hadamard/phase_a/origin_control/origin-control-lib.sh
. "$SCRIPT_DIR/origin-control-lib.sh"

[ "$#" -eq 0 ] || { origin_die "validate-origin-control.sh takes no arguments"; exit 2; }
origin_preflight_stage

for script in "$ORIGIN_MARKER" "$ORIGIN_SUBMIT" "$ORIGIN_CONDITION" "$ORIGIN_ACTION" \
              "$ORIGIN_STAGE_ROOT/$ORIGIN_VALIDATE_RELATIVE" \
              "$ORIGIN_STAGE_ROOT/$ORIGIN_LIBRARY_RELATIVE"; do
  bash -n "$script"
done

actions=$(awk '
  NR == 1 && /^#!/ { next }
  /^[[:space:]]*($|#)/ { next }
  { print }
' "$ORIGIN_MARKER")
expected=$(printf '%s\n%s' \
  "printf 'token=origin-control-marker-r1-6c6043979a4a4a32 slurm_job_id=%s hostname=%s\\n' \"\$SLURM_JOB_ID\" \"\$HOSTNAME\"" \
  'exit 0')
[ "$actions" = "$expected" ] || { origin_die "marker executable action order drifted"; exit 2; }
if grep -Eiq 'python|cuda|triton|nvidia-smi|model|import|benchmark|scientific' "$ORIGIN_MARKER"; then
  origin_die "marker contains a forbidden payload reference"
  exit 2
fi

temporary=$(mktemp -d "$ORIGIN_RESULT_ROOT/.validation.XXXXXX")
trap 'rm -rf -- "$temporary"' EXIT
declare -a digests=()
for lane in direct main-crew secondmate-crew; do
  origin_validate_lane "$lane"
  origin_build_sbatch_argv
  printf '%s\0' "${ORIGIN_SBATCH_ARGV[@]}" > "$temporary/$lane.argv"
  digests+=("$(origin_sha256 "$temporary/$lane.argv")")
done
[ "${digests[0]}" = "${digests[1]}" ] && [ "${digests[1]}" = "${digests[2]}" ] \
  || { origin_die "lane sbatch argument vectors are not byte-equivalent"; exit 2; }
if origin_validate_lane invalid-lane >/dev/null 2>&1; then
  origin_die "invalid lane was accepted"
  exit 2
fi

saved_result_root=$ORIGIN_RESULT_ROOT
ORIGIN_RESULT_ROOT="$temporary/one-shot"
mkdir -m 0700 "$ORIGIN_RESULT_ROOT"
mkdir -m 0700 "$ORIGIN_RESULT_ROOT/latches" "$ORIGIN_RESULT_ROOT/ledgers"
origin_claim_lane_once direct >/dev/null
if origin_claim_lane_once direct >/dev/null 2>&1; then
  origin_die "second lane claim was not refused"
  exit 2
fi
ORIGIN_RESULT_ROOT=$saved_result_root

printf 'validated=true\n'
printf 'stage=%s\n' "$ORIGIN_STAGE_ROOT"
printf 'result_root=%s\n' "$ORIGIN_RESULT_ROOT"
printf 'marker_sha256=%s\n' "$(origin_sha256 "$ORIGIN_MARKER")"
printf 'submit_sha256=%s\n' "$(origin_sha256 "$ORIGIN_SUBMIT")"
printf 'sbatch_argv_sha256=%s\n' "${digests[0]}"
printf 'lane_vectors=byte-equivalent\n'
printf 'one_shot_refusal=passed\n'
printf 'submission_performed=false\n'
