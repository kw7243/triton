#!/usr/bin/env bash
set -euo pipefail
umask 077

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=experiments/structured_hadamard/phase_a/origin_control/origin-control-lib.sh
. "$SCRIPT_DIR/origin-control-lib.sh"

[ "$#" -eq 1 ] || { origin_die "usage: submit-origin-once.sh LANE"; exit 2; }
lane=$1
origin_validate_lane "$lane" || exit 2
origin_preflight_stage || exit 2
origin_assert_lane_fresh "$lane" || exit 2
origin_build_sbatch_argv
origin_build_terminal_owner_argv __JOB_ID__
[ -x /usr/bin/sbatch ] || { origin_die "/usr/bin/sbatch is unavailable"; exit 2; }
[ "$PWD" = "$ORIGIN_STAGE_ROOT" ] || cd "$ORIGIN_STAGE_ROOT"

latch=$(origin_claim_lane_once "$lane") || exit 2
{
  printf 'schema=origin-submit-latch-v1\n'
  printf 'lane=%s\n' "$lane"
  printf 'attempt_budget=1\n'
  printf 'stage=%s\n' "$ORIGIN_STAGE_ROOT"
  printf 'result_root=%s\n' "$ORIGIN_RESULT_ROOT"
  printf 'cwd=%s\n' "$PWD"
  printf 'marker=%s\n' "$ORIGIN_MARKER"
  printf 'marker_sha256=%s\n' "$(origin_sha256 "$ORIGIN_MARKER")"
  printf 'created_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'submit_order=validate,handoff,claim,record-attempt,sbatch,record-response\n'
} > "$latch/preflight.txt"
printf '%s\0' "${ORIGIN_SBATCH_ARGV[@]}" > "$latch/sbatch.argv.nul"
printf '%s\0' "${ORIGIN_TERMINAL_OWNER_ARGV[@]}" > "$latch/terminal-owner-template.argv.nul"
printf '1\n' > "$latch/sbatch-attempt-count"
chmod 0400 "$latch/preflight.txt" "$latch/sbatch.argv.nul" \
  "$latch/terminal-owner-template.argv.nul" "$latch/sbatch-attempt-count"
sync "$latch/sbatch-attempt-count" 2>/dev/null || true

set +e
"${ORIGIN_SBATCH_ARGV[@]}" > "$latch/sbatch.stdout" 2> "$latch/sbatch.stderr"
sbatch_exit=$?
set -e
printf '%s\n' "$sbatch_exit" > "$latch/sbatch.exit-code"
chmod 0400 "$latch/sbatch.stdout" "$latch/sbatch.stderr" "$latch/sbatch.exit-code"

job_id=$(awk 'NF { print; count++ } END { if (count != 1) exit 1 }' "$latch/sbatch.stdout") \
  || { origin_die "sole sbatch response was ambiguous; lane attempt is consumed and will not be retried"; exit 3; }
if [ "$sbatch_exit" -ne 0 ] || ! origin_validate_job_id "$job_id"; then
  origin_die "sole sbatch attempt failed or returned a noncanonical job id; lane attempt is consumed and will not be retried"
  exit 3
fi

origin_build_terminal_owner_argv "$job_id"
printf '%s\0' "${ORIGIN_TERMINAL_OWNER_ARGV[@]}" > "$latch/terminal-owner.argv.nul"
chmod 0400 "$latch/terminal-owner.argv.nul"
ledger="$ORIGIN_RESULT_ROOT/ledgers/$lane.txt"
{
  printf 'schema=origin-submission-ledger-v1\n'
  printf 'lane=%s\n' "$lane"
  printf 'sbatch_attempts=1\n'
  printf 'sbatch_exit_code=%s\n' "$sbatch_exit"
  printf 'job_id=%s\n' "$job_id"
  printf 'submitted_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'stage=%s\n' "$ORIGIN_STAGE_ROOT"
  printf 'cwd=%s\n' "$PWD"
  printf 'marker=%s\n' "$ORIGIN_MARKER"
  printf 'output_template=%s\n' "$ORIGIN_OUTPUT_TEMPLATE"
  printf 'raw_stdout=%s\n' "$latch/sbatch.stdout"
  printf 'raw_stderr=%s\n' "$latch/sbatch.stderr"
  printf 'terminal_owner=when-slurm-%s\n' "$job_id"
} > "$ledger"
chmod 0400 "$ledger"
printf 'job_id=%s\nraw_stdout=%s\nraw_stderr=%s\nledger=%s\nterminal_owner=when-slurm-%s\n' \
  "$job_id" "$latch/sbatch.stdout" "$latch/sbatch.stderr" "$ledger" "$job_id"
