#!/usr/bin/env python3
"""Exactly-once salloc/srun owner for the real-model Phase A comparison."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

if __package__:
    from .preflight_remote_audit import (
        RESULT_SCHEMA_VERSION,
        audit,
        sha256_file,
    )
else:
    from preflight_remote_audit import (  # type: ignore[no-redef]
        RESULT_SCHEMA_VERSION,
        audit,
        sha256_file,
    )


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _exclusive_json(path: Path, value: object) -> None:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _append_status(path: Path, event: str, **values: object) -> None:
    payload = {"recorded_at_utc": _utc(), "event": event, **values}
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
    with os.fdopen(descriptor, "ab") as stream:
        stream.write((json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def _advance(path: Path, key: str) -> dict[str, int]:
    value = _load(path)
    if value.get(key) != 0:
        raise RuntimeError(f"one-shot ledger refusal: {key}={value.get(key)!r}")
    value[key] = 1
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    _exclusive_json(temporary, value)
    os.replace(temporary, path)
    return value


def _helper_record(clearance: dict[str, object], name: str) -> dict[str, object]:
    matches = [record for record in clearance["helpers"] if record["name"] == name]
    if len(matches) != 1:
        raise RuntimeError(f"clearance must bind exactly one helper named {name}")
    return matches[0]


def _verify_preflight(clearance_path: Path, clearance: dict[str, object]) -> None:
    helper = _helper_record(clearance, "preflight_remote_audit")
    output_path = Path(clearance["audit_output"])
    output = _load(output_path)
    clearance_sha256 = sha256_file(clearance_path)
    if (output.get("schema_version") != RESULT_SCHEMA_VERSION or output.get("status") != "passed"
            or output.get("helper", {}).get("sha256") != helper["sha256"]
            or output.get("clearance", {}).get("sha256") != clearance_sha256):
        raise RuntimeError("preserved staged preflight result is absent or differs")
    audit(
        clearance_path, expected_clearance_sha256=clearance_sha256,
        expected_helper_sha256=helper["sha256"], allow_existing_output=True,
    )


def _allocated(clearance_path: Path) -> int:
    clearance = _load(clearance_path)
    ledger = Path(clearance["ledger"]["path"])
    status_path = Path(clearance["ledger"]["status_path"])
    job_id = os.environ.get("SLURM_JOB_ID")
    partition = os.environ.get("SLURM_JOB_PARTITION")
    if not job_id or not partition:
        raise RuntimeError("allocated owner lacks Slurm job id or partition")
    _append_status(status_path, "allocation_acquired", job_id=job_id, partition=partition)
    _advance(ledger, "srun_attempts")
    _advance(ledger, "driver_attempts")
    argv = [*clearance["execution"]["srun_argv"], *clearance["execution"]["driver_argv"]]
    environment = {**os.environ, **clearance["environment"]["runtime_env"]}
    completed = subprocess.run(argv, check=False, env=environment)
    if completed.returncode:
        _append_status(
            status_path, "launch_or_runtime_error", job_id=job_id, partition=partition,
            srun_exit=completed.returncode,
        )
    return completed.returncode


def _owner(clearance_path: Path) -> int:
    clearance = _load(clearance_path)
    ledger = Path(clearance["ledger"]["path"])
    status_path = Path(clearance["ledger"]["status_path"])
    terminal_path = Path(clearance["terminal_path"])
    owner_exit = 125
    failure = None
    try:
        _verify_preflight(clearance_path, clearance)
        _advance(ledger, "owner_attempts")
        _advance(ledger, "salloc_attempts")
        _append_status(status_path, "scheduler_owner_started")
        argv = [*clearance["execution"]["salloc_argv"], *clearance["execution"]["allocated_argv"]]
        environment = {**os.environ, **clearance["environment"]["runtime_env"]}
        owner_exit = subprocess.run(argv, check=False, env=environment).returncode
        if owner_exit:
            failure = f"salloc_owner_exit={owner_exit}"
        return owner_exit
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        _append_status(status_path, "launch_or_runtime_error", error=failure)
        raise
    finally:
        terminal_ledger = _advance(ledger, "terminal_events_fired")
        terminal = {
            "schema_version": "phase-a-real-model-terminal-event-v1",
            "recorded_at_utc": _utc(), "owner_exit": owner_exit,
            "failure": failure, "ledger": terminal_ledger,
        }
        _exclusive_json(terminal_path, terminal)
        _append_status(status_path, "terminal_state", **terminal)
        print("TERMINAL " + json.dumps(terminal, sort_keys=True), flush=True)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) != 2 or argv[0] not in {"--owner", "--allocated"}:
        raise SystemExit("usage: gpu_owner.py (--owner|--allocated) CLEARANCE.json")
    clearance = Path(argv[1]).resolve(strict=True)
    return _owner(clearance) if argv[0] == "--owner" else _allocated(clearance)


if __name__ == "__main__":
    raise SystemExit(main())
