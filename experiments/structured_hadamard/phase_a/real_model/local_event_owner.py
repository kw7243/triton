#!/usr/bin/env python3
"""Local tmux-owned SSH runner that emits the task's sole terminal signal."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh", required=True)
    parser.add_argument("--control-socket", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--terminal-record", type=Path, required=True)
    parser.add_argument("--tmux-event", required=True)
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
    try:
        with args.log.open("xb") as stream:
            completed = subprocess.run(
                build_ssh_argv(args), check=False, stdout=stream, stderr=subprocess.STDOUT,
            )
        exit_code = completed.returncode
        if exit_code:
            failure = f"ssh_owner_exit={exit_code}"
        return exit_code
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        terminal = {
            "schema_version": "phase-a-real-model-local-terminal-v1",
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "exit_code": exit_code, "failure": failure,
            "ssh_argv": build_ssh_argv(args),
        }
        _exclusive_json(args.terminal_record, terminal)
        subprocess.run(("tmux", "wait-for", "-S", args.tmux_event), check=False)


if __name__ == "__main__":
    raise SystemExit(main())
