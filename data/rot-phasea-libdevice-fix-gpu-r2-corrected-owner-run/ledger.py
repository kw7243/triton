#!/usr/bin/env python3
"""Atomically move one corrected-attempt ledger counter from zero to one."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


path = Path(sys.argv[1])
key = sys.argv[2]
value = json.loads(path.read_text(encoding="utf-8"))
if key not in value or value[key] != 0:
    raise SystemExit(f"ledger refusal: {key} must be exactly zero, got {value.get(key)!r}")
value[key] = 1
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
    json.dump(value, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, path)
