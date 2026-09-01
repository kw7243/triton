#!/usr/bin/env bash
set -uo pipefail
umask 077

attempt=/data/scratch-fast/kwen1/structured-hadamard/rot-phasea-libdevice-fix-gpu-r2-corrected-owner-run/attempt-20260901T214700Z-48a972220979
run_state=$attempt/run-state
printf 'remote_owner_started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cd "$run_state" || exit 125
sha256sum -c helper-sha256sums.txt || exit 125
"$run_state/allocation-owner.sh"
attempt_rc=$?
"$run_state/ledger.py" "$run_state/ledgers.json" terminal_events_fired
PYTHONDONTWRITEBYTECODE=1 /data/scratch-fast/kwen1/micromamba/root/envs/causal_forcing/bin/python \
  - "$run_state/remote-terminal.json" "$run_state/ledgers.json" "$attempt_rc" <<'PY'
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

path = Path(sys.argv[1])
ledger_path = Path(sys.argv[2])
value = {
    "schema_version": "rot-phasea-corrected-terminal-event-v1",
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "remote_attempt_exit": int(sys.argv[3]),
    "ledgers": json.loads(ledger_path.read_text(encoding="utf-8")),
}
encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "wb") as stream:
    stream.write(encoded)
    stream.flush()
    os.fsync(stream.fileno())
print(json.dumps(value, sort_keys=True))
PY
printf 'remote_owner_finished_utc=%s exit=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$attempt_rc"
exit "$attempt_rc"
