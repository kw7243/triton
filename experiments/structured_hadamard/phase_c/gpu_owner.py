#!/usr/bin/env python3
"""Exactly-once salloc/srun owner for the frozen Phase C selector run."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.structured_hadamard.phase_c.preflight import (
    RESULT_SCHEMA_VERSION,
    audit,
)
from experiments.structured_hadamard.phase_c.selector import sha256_file


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


def _status(path: Path, event: str, **values: object) -> None:
    payload = {"recorded_at_utc": _utc(), "event": event, **values}
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
    with os.fdopen(descriptor, "ab") as stream:
        stream.write((json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    print("STATUS " + json.dumps(payload, sort_keys=True), flush=True)


def _advance(path: Path, key: str) -> dict[str, int]:
    value = _load(path)
    if value.get(key) != 0:
        raise RuntimeError(f"one-shot ledger refusal: {key}={value.get(key)!r}")
    value[key] = 1
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    _exclusive_json(temporary, value)
    os.replace(temporary, path)
    return value


def _helper(clearance: MappingLike, name: str) -> dict[str, object]:
    matches = [record for record in clearance["helpers"] if record["name"] == name]
    if len(matches) != 1:
        raise RuntimeError(f"clearance must bind exactly one helper named {name}")
    return matches[0]


MappingLike = dict[str, object]


def _verify_preflight(clearance_path: Path, clearance: MappingLike) -> None:
    helper = _helper(clearance, "phase_c_preflight")
    output_path = Path(clearance["audit_output"])
    output = _load(output_path)
    clearance_sha256 = sha256_file(clearance_path)
    if (output.get("schema_version") != RESULT_SCHEMA_VERSION or output.get("status") != "passed"
            or output.get("helper", {}).get("sha256") != helper["sha256"]
            or output.get("clearance", {}).get("sha256") != clearance_sha256):
        raise RuntimeError("preserved staged Phase C preflight result is absent or differs")
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
    _status(status_path, "allocation_acquired", job_id=job_id, partition=partition)
    _advance(ledger, "srun_attempts")
    _advance(ledger, "driver_attempts")
    argv = [*clearance["execution"]["srun_argv"], *clearance["execution"]["driver_argv"]]
    environment = {**os.environ, **clearance["environment"]["runtime_env"]}
    completed = subprocess.run(argv, check=False, env=environment)
    if completed.returncode:
        _status(status_path, "launch_or_runtime_error", job_id=job_id, partition=partition,
                srun_exit=completed.returncode)
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
        _status(status_path, "scheduler_owner_started")
        argv = [*clearance["execution"]["salloc_argv"], *clearance["execution"]["allocated_argv"]]
        environment = {**os.environ, **clearance["environment"]["runtime_env"]}
        owner_exit = subprocess.run(argv, check=False, env=environment).returncode
        if owner_exit:
            failure = f"salloc_owner_exit={owner_exit}"
        return owner_exit
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        _status(status_path, "launch_or_runtime_error", error=failure)
        raise
    finally:
        terminal_ledger = _advance(ledger, "terminal_events_fired")
        terminal = {
            "schema_version": "phase-c-selector-terminal-event-v1",
            "recorded_at_utc": _utc(), "owner_exit": owner_exit,
            "failure": failure, "ledger": terminal_ledger,
        }
        _exclusive_json(terminal_path, terminal)
        _status(status_path, "terminal_state", **terminal)
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
