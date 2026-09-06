#!/usr/bin/env python3
"""Consume one immutable timing request with at most one sbatch invocation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inventory import build, file_sha256


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_ledger(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text().splitlines()]
    previous = "0" * 64
    for sequence, record in enumerate(records):
        core = {key: value for key, value in record.items() if key != "record_sha256"}
        digest = hashlib.sha256(
            json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if (
            record.get("sequence") != sequence
            or record.get("previous_sha256") != previous
            or record.get("record_sha256") != digest
        ):
            raise RuntimeError("attempt ledger hash chain is invalid")
        previous = digest
    return records


def append(path: Path, event: str, payload: dict[str, Any]) -> None:
    records = load_ledger(path)
    previous = records[-1]["record_sha256"]
    core = {
        "schema": "vq-phase-a-attempt-ledger/v1",
        "sequence": len(records),
        "recorded_at_utc": utc_now(),
        "event": event,
        "previous_sha256": previous,
        "payload": payload,
    }
    record = {
        **core,
        "record_sha256": hashlib.sha256(
            json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
    with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    manifest_path = Path(request["launch_manifest"]).resolve(strict=True)
    digest = file_sha256(manifest_path)
    if digest != request["launch_manifest_sha256"]:
        raise RuntimeError("submission request manifest digest mismatch")
    records = load_ledger(args.ledger)
    if file_sha256(args.request) != records[0]["payload"]["submission_request_sha256"]:
        raise RuntimeError("submission request bytes differ from prepared ledger")
    if len(records) != 1 or records[0]["event"] != "prepared":
        raise RuntimeError("submission budget is already consumed")
    if any(name.startswith("SBATCH_") for name in os.environ):
        raise RuntimeError("ambient SBATCH_* variables are forbidden")
    exports = [
        argument for argument in request["sbatch_argv"] if argument.startswith("--export")
    ]
    if exports != ["--export=NIL"]:
        raise RuntimeError("submission must use exactly literal --export=NIL")
    if request.get("export_policy") != (
        "literal --export=NIL; required values assigned in script"
    ):
        raise RuntimeError("submission export policy mismatch")
    if request.get("launch_route") != "batch-primary":
        raise RuntimeError("submission route mismatch")
    stage = Path(request["cwd"]).resolve(strict=True)
    if Path.cwd().resolve(strict=True) != stage:
        raise RuntimeError("submission must run from the frozen stage")
    manifest = json.loads(manifest_path.read_text())
    current = build(stage)
    expected = manifest["stage_inventory"]
    if (
        current["content_digest"] != expected["content_digest"]
        or current["record_count"] != expected["record_count"]
    ):
        raise RuntimeError("frozen stage changed before submission")
    prepared = records[0]["payload"]
    if prepared["sbatch_invocations"] != 0 or prepared["accepted_job_id"] is not None:
        raise RuntimeError("prepared ledger already records a submission")
    append(
        args.ledger,
        "route_selected",
        {
            "launch_route": "batch-primary",
            "launch_manifest_sha256": digest,
            "submission_request_sha256": file_sha256(args.request),
            "sbatch_invocations": 1,
            "accepted_job_id": None,
            "sbatch_argv": request["sbatch_argv"],
        },
    )
    completed = subprocess.run(
        request["sbatch_argv"],
        cwd=stage,
        env=dict(os.environ),
        text=True,
        capture_output=True,
        check=False,
    )
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode != 0:
        append(
            args.ledger,
            "submission_failed",
            {
                "launch_manifest_sha256": digest,
                "returncode": completed.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "retry_forbidden": True,
            },
        )
        raise RuntimeError(
            f"sole sbatch invocation failed with {completed.returncode}; retry forbidden"
        )
    match = re.fullmatch(r"([0-9]+)(?:;[^\s;]+)?", stdout)
    if match is None:
        append(
            args.ledger,
            "submission_unparseable",
            {
                "launch_manifest_sha256": digest,
                "returncode": completed.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "retry_forbidden": True,
            },
        )
        raise RuntimeError("sole sbatch response is nonnumeric; retry forbidden")
    job_id = match.group(1)
    append(
        args.ledger,
        "accepted",
        {
            "launch_manifest_sha256": digest,
            "sbatch_invocations": 1,
            "accepted_job_id": job_id,
            "stdout": stdout,
            "stderr": stderr,
            "retry_forbidden": True,
        },
    )
    print(job_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
