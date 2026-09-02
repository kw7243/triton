#!/usr/bin/env python3
"""Single local tmux/SSH terminal owner with event-driven Firstmate status relay."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess


def build_ssh_argv(args: argparse.Namespace) -> list[str]:
    return [
        args.ssh, "-S", str(args.control_socket), "-o", "ControlMaster=no",
        "-o", "BatchMode=yes", "-tt", args.host, *args.remote_argv,
    ]


def _exclusive_json(path: Path, value: object) -> None:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _append_firstmate(path: Path, line: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
    with os.fdopen(descriptor, "ab", closefd=True) as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write((line.rstrip("\n") + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _summary(key: str, payload: dict[str, object]) -> str | None:
    event = payload.get("event")
    prefix = f"working [key={key}]:"
    if event == "scheduler_owner_started":
        return f"{prefix} sole Phase B scheduler owner started; one-shot salloc ledger consumed"
    if event == "allocation_acquired":
        return f"{prefix} allocation job={payload.get('job_id')} partition={payload.get('partition')}"
    if event == "scientific_driver_started":
        return f"{prefix} sole scientific driver launched; output={payload.get('output_directory')}"
    if event == "gpu_visible":
        hardware = payload.get("hardware", {})
        return (f"{prefix} GPU visible job={hardware.get('job_id')} partition={hardware.get('partition')} "
                f"gpu={hardware.get('name')} capability={hardware.get('compute_capability')}")
    if event in {"runtime_error", "launch_or_runtime_error"}:
        return f"blocked [key={key}]: Phase B launch/runtime issue: {payload.get('error') or payload.get('srun_exit')}"
    if event == "scientific_driver_completed":
        return f"{prefix} scientific driver completed; {payload.get('decision')}"
    if event == "terminal_state":
        return (f"{prefix} terminal outcome owner_exit={payload.get('owner_exit')} "
                f"failure={payload.get('failure')} ledger={payload.get('ledger')}")
    return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh", required=True)
    parser.add_argument("--control-socket", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--terminal-record", type=Path, required=True)
    parser.add_argument("--tmux-event", required=True)
    parser.add_argument("--firstmate-status", type=Path, required=True)
    parser.add_argument("--status-key", default="phaseb-map-r1")
    parser.add_argument("remote_argv", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.remote_argv[:1] == ["--"]:
        args.remote_argv = args.remote_argv[1:]
    if not args.remote_argv:
        raise SystemExit("remote owner argv is required after --")
    exit_code = 125
    failure = None
    ssh_argv = build_ssh_argv(args)
    try:
        with args.log.open("xb") as log:
            process = subprocess.Popen(
                ssh_argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line.encode("utf-8", errors="replace"))
                log.flush()
                if line.startswith("STATUS "):
                    try:
                        payload = json.loads(line[len("STATUS "):])
                        summary = _summary(args.status_key, payload)
                        if summary:
                            _append_firstmate(args.firstmate_status, summary)
                    except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
                        _append_firstmate(
                            args.firstmate_status,
                            f"blocked [key={args.status_key}]: local terminal owner could not relay status: {type(error).__name__}",
                        )
            exit_code = process.wait()
        if exit_code:
            failure = f"ssh_owner_exit={exit_code}"
        return exit_code
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        terminal = {
            "schema_version": "phase-b-map-local-terminal-v1",
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "exit_code": exit_code, "failure": failure, "ssh_argv": ssh_argv,
        }
        _exclusive_json(args.terminal_record, terminal)
        subprocess.run(("tmux", "wait-for", "-S", args.tmux_event), check=False)


if __name__ == "__main__":
    raise SystemExit(main())
