#!/usr/bin/env bash
set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P) || exit 2
# shellcheck source=experiments/structured_hadamard/phase_a/origin_control/origin-control-lib.sh
. "$SCRIPT_DIR/origin-control-lib.sh"

[ "$#" -eq 1 ] || { origin_die "usage: terminal-condition.sh JOB_ID"; exit 2; }
origin_validate_job_id "$1" || exit 2

row=$(/usr/bin/sacct -X -n -P -j "$1" -o JobIDRaw,State 2>/dev/null) || exit 2
state=$(awk -F'|' -v job="$1" '$1 == job { print $2; count++ } END { if (count != 1) exit 1 }' <<<"$row") || exit 1
state=${state%% *}
state=${state%%+}
origin_terminal_state "$state"
