#!/usr/bin/env bash
set -euo pipefail
umask 077

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=experiments/structured_hadamard/phase_a/origin_control/origin-control-lib.sh
. "$SCRIPT_DIR/origin-control-lib.sh"

[ "$#" -eq 1 ] || { origin_die "usage: terminal-action.sh JOB_ID"; exit 2; }
job_id=$1
origin_validate_job_id "$job_id" || exit 2
origin_resolve_layout || exit 2

latch="$ORIGIN_RESULT_ROOT/terminal-latches/$job_id"
capture="$ORIGIN_RESULT_ROOT/terminal-captures/$job_id"
[ ! -e "$capture" ] && [ ! -L "$capture" ] || { origin_die "terminal capture already exists"; exit 2; }
mkdir "$latch" 2>/dev/null || { origin_die "terminal capture was already claimed"; exit 2; }
printf 'job_id=%s\nclaimed_utc=%s\n' "$job_id" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$latch/claim.txt"
chmod 0400 "$latch/claim.txt"

mapfile -t matching_ledgers < <(grep -l -x "job_id=$job_id" "$ORIGIN_RESULT_ROOT"/ledgers/*.txt 2>/dev/null || true)
[ "${#matching_ledgers[@]}" -eq 1 ] || { origin_die "job id does not map to exactly one experiment ledger"; exit 2; }
lane=${matching_ledgers[0]##*/}
lane=${lane%.txt}
origin_validate_lane "$lane" || exit 2

temporary=$(mktemp -d "$ORIGIN_RESULT_ROOT/terminal-captures/.capture-$job_id.XXXXXX")
raw="$temporary/accounting.raw"
/usr/bin/sacct -X -n -P -j "$job_id" \
  -o JobIDRaw,JobName,State,Reason,ExitCode,DerivedExitCode,NodeList,Submit,Eligible,Start,End,Elapsed,Account,QOS,Partition,NNodes,NTasks,AllocCPUS,ReqMem,Timelimit,ReqTRES,AllocTRES,WorkDir,StdOut,StdErr \
  > "$raw"
row=$(awk -F'|' -v job="$job_id" '$1 == job { print; count++ } END { if (count != 1) exit 1 }' "$raw") \
  || { origin_die "terminal accounting did not contain one exact allocation row"; exit 2; }
state=$(awk -F'|' '{print $3}' <<<"$row")
terminal_state=${state%% *}
terminal_state=${terminal_state%%+}
origin_terminal_state "$terminal_state" || { origin_die "action observed a nonterminal state"; exit 2; }

output="$ORIGIN_RESULT_ROOT/slurm-$job_id.out"
{
  printf 'job_id=%s\n' "$job_id"
  printf 'lane=%s\n' "$lane"
  printf 'terminal_state=%s\n' "$terminal_state"
  printf 'captured_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'stage=%s\n' "$ORIGIN_STAGE_ROOT"
  printf 'output_path=%s\n' "$output"
  printf 'sacct_row=%s\n' "$row"
  if [ -f "$output" ] && [ ! -L "$output" ]; then
    cp -- "$output" "$temporary/slurm-$job_id.out.capture"
    printf 'output_exists=true\n'
    printf 'output_size=%s\n' "$(stat -c '%s' "$temporary/slurm-$job_id.out.capture")"
    printf 'output_sha256=%s\n' "$(origin_sha256 "$temporary/slurm-$job_id.out.capture")"
  else
    printf 'output_exists=false\n'
  fi
} > "$temporary/summary.txt"
chmod 0600 "$raw" "$temporary/summary.txt" "$temporary"/slurm-*.out.capture 2>/dev/null || true
mv -- "$temporary" "$capture"
chmod 0500 "$capture"
cat "$capture/summary.txt"
