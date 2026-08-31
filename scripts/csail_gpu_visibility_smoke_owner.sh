#!/usr/bin/env bash
set -euo pipefail

umask 077

gate_name=${SMOKE_LOCAL_GATE_NAME:?SMOKE_LOCAL_GATE_NAME is required}
session=${SMOKE_TMUX_SESSION:?SMOKE_TMUX_SESSION is required}
terminal_marker=${SMOKE_TERMINAL_MARKER:?SMOKE_TERMINAL_MARKER is required}
remote_host=${SMOKE_REMOTE_HOST:?SMOKE_REMOTE_HOST is required}
remote_launcher=${SMOKE_REMOTE_LAUNCHER:?SMOKE_REMOTE_LAUNCHER is required}
attempt=${SMOKE_ATTEMPT_ROOT:?SMOKE_ATTEMPT_ROOT is required}
stage=${SMOKE_STAGE_PATH:?SMOKE_STAGE_PATH is required}
result_dir=${SMOKE_RESULT_DIR:?SMOKE_RESULT_DIR is required}
expected_head=${SMOKE_EXPECTED_HEAD:?SMOKE_EXPECTED_HEAD is required}
expected_metadata_sha=${SMOKE_EXPECTED_METADATA_SHA256:?SMOKE_EXPECTED_METADATA_SHA256 is required}
partition=${SMOKE_EXPECTED_PARTITION:?SMOKE_EXPECTED_PARTITION is required}
account=${SMOKE_EXPECTED_ACCOUNT:?SMOKE_EXPECTED_ACCOUNT is required}
qos=${SMOKE_EXPECTED_QOS:?SMOKE_EXPECTED_QOS is required}

[ ! -e "$terminal_marker" ] && [ ! -L "$terminal_marker" ]
/usr/bin/tmux wait-for "$gate_name"

set +e
/usr/bin/ssh \
  -tt \
  -oBatchMode=yes \
  -oConnectTimeout=15 \
  "$remote_host" \
  /usr/bin/env \
  "SMOKE_ATTEMPT_ROOT=$attempt" \
  "SMOKE_STAGE_PATH=$stage" \
  "SMOKE_RESULT_DIR=$result_dir" \
  "SMOKE_EXPECTED_HEAD=$expected_head" \
  "SMOKE_EXPECTED_METADATA_SHA256=$expected_metadata_sha" \
  "SMOKE_EXPECTED_PARTITION=$partition" \
  "SMOKE_EXPECTED_ACCOUNT=$account" \
  "SMOKE_EXPECTED_QOS=$qos" \
  /bin/bash "$remote_launcher"
ssh_rc=$?
set -e

marker_tmp=${terminal_marker}.$$.tmp
{
  printf 'schema=tmux-ssh-terminal-v1\n'
  printf 'session=%s\n' "$session"
  printf 'ssh_exit_code=%s\n' "$ssh_rc"
  printf 'exited_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$marker_tmp"
chmod 0600 "$marker_tmp"
mv -- "$marker_tmp" "$terminal_marker"
exit "$ssh_rc"
