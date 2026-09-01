#!/usr/bin/env bash
set -uo pipefail
umask 077

session=rot-phasea-corrected-owner
event_token=rot-phasea-corrected-terminal
local_event=/tmp/rot-phasea-corrected-control.Umqlas/terminal-event.json
control_socket=/tmp/rot-phasea-corrected-control.Umqlas/ssh-control
remote_script=/data/scratch-fast/kwen1/structured-hadamard/rot-phasea-libdevice-fix-gpu-r2-corrected-owner-run/attempt-20260901T214700Z-48a972220979/run-state/remote-attempt.sh

finalize() {
  owner_rc=$?
  trap - EXIT
  event_tmp=${local_event}.tmp.$$
  printf '{"schema_version":"rot-phasea-corrected-local-terminal-event-v1","recorded_at_utc":"%s","tmux_session":"%s","ssh_host":"slurm-login.csail.mit.edu","ssh_exit":%s,"remote_terminal":"%s"}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$session" "$owner_rc" \
    "/data/scratch-fast/kwen1/structured-hadamard/rot-phasea-libdevice-fix-gpu-r2-corrected-owner-run/attempt-20260901T214700Z-48a972220979/run-state/remote-terminal.json" \
    > "$event_tmp"
  chmod 600 "$event_tmp"
  mv "$event_tmp" "$local_event"
  tmux wait-for -S "$event_token"
  exit "$owner_rc"
}
trap finalize EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

/usr/bin/ssh -S "$control_socket" -o ControlMaster=no -o BatchMode=yes -tt \
  slurm-login.csail.mit.edu "$remote_script"
