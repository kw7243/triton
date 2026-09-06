#!/usr/bin/env python3
"""Validate the terminal accounting and immutable timing artifact set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inventory import build, file_sha256
from runtime import atomic_json


TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "TIMEOUT",
}


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def safe_object(path: Path, label: str, errors: list[str]) -> dict[str, Any]:
    try:
        return read_object(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"{label} unavailable: {exc}")
        return {}


def ledger(path: Path) -> list[dict[str, Any]]:
    values = [json.loads(line) for line in path.read_text().splitlines()]
    previous = "0" * 64
    for sequence, record in enumerate(values):
        core = {key: value for key, value in record.items() if key != "record_sha256"}
        digest = hashlib.sha256(
            json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if (
            record.get("sequence") != sequence
            or record.get("previous_sha256") != previous
            or record.get("record_sha256") != digest
        ):
            raise ValueError("attempt ledger hash chain is invalid")
        previous = digest
    return values


def accounting(path: Path, job_id: str) -> tuple[dict[str, str], list[str]]:
    errors: list[str] = []
    rows = [
        line.split("|")
        for line in path.read_text().splitlines()
        if line and line.split("|", 1)[0] == job_id
    ]
    if len(rows) != 1:
        return {}, [f"expected one root accounting row, found {len(rows)}"]
    fields = (
        "job_id",
        "job_name",
        "state",
        "exit_code",
        "derived_exit_code",
        "submit",
        "start",
        "end",
        "elapsed",
        "node_list",
        "partition",
        "account",
        "qos",
        "alloc_cpus",
        "req_mem",
        "alloc_tres",
    )
    if len(rows[0]) != len(fields):
        return {}, [f"accounting field count {len(rows[0])} != {len(fields)}"]
    value = dict(zip(fields, rows[0], strict=True))
    state = value["state"].split()[0].split("+")[0]
    value["normalized_state"] = state
    if state not in TERMINAL_STATES:
        errors.append(f"accounting state is not terminal: {value['state']}")
    if not value["end"] or value["end"] == "Unknown":
        errors.append("accounting end time is empty")
    return value, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--accounting", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.result.resolve(strict=True)
    errors: list[str] = []
    if re.fullmatch(r"[0-9]+", args.job_id) is None:
        errors.append("job ID is not numeric")
    manifest_path = root / "launch_manifest.json"
    try:
        if file_sha256(manifest_path) != args.manifest_sha256:
            errors.append("launch manifest SHA-256 mismatch")
    except OSError as exc:
        errors.append(f"launch manifest unavailable: {exc}")
    manifest = safe_object(manifest_path, "launch manifest", errors)
    final = safe_object(root / "final_result_manifest.json", "final result", errors)
    timing = (
        safe_object(root / "timing_output.json", "timing output", errors)
        if (root / "timing_output.json").exists()
        else {}
    )
    runtime = safe_object(root / "runtime_start.json", "runtime start", errors)
    environment = safe_object(
        root / "environment_validation.json", "environment validation", errors
    )
    for name, value in (
        ("final result", final),
        ("runtime start", runtime),
        ("environment validation", environment),
    ):
        if value.get("launch_manifest_sha256") != args.manifest_sha256:
            errors.append(f"{name} manifest binding mismatch")
    if timing and timing.get("launch_manifest_sha256") != args.manifest_sha256:
        errors.append("timing output manifest binding mismatch")
    if final.get("job_id") != args.job_id or runtime.get("job_id") != args.job_id:
        errors.append("runtime/final job ID mismatch")
    if environment.get("status") != "passed" and final.get("classification") != "NO RESULT":
        errors.append("scientific classification exists without environment PASS")
    for name, expected in final.get("files", {}).items():
        path = root / name
        if not path.is_file() or path.is_symlink():
            errors.append(f"final-manifest artifact unavailable: {name}")
        elif file_sha256(path) != expected.get("sha256"):
            errors.append(f"final-manifest artifact changed: {name}")
    required = {
        "timing_output.json",
        "trial_timings.json",
        "timings.csv",
        "table.md",
        "latency.png",
        "run_metadata.json",
    }
    if final.get("classification") != "NO RESULT":
        missing = sorted(name for name in required if not (root / name).is_file())
        if missing:
            errors.append(f"scientific result artifacts missing: {missing}")
    try:
        values = ledger(root / "attempt_ledger.jsonl")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"attempt ledger invalid: {exc}")
        values = []
    accepted = [value for value in values if value.get("event") == "accepted"]
    started = [value for value in values if value.get("event") == "submission_started"]
    if len(started) != 1 or len(accepted) != 1:
        errors.append("ledger does not contain exactly one submission and acceptance")
    elif accepted[0].get("payload", {}).get("accepted_job_id") != args.job_id:
        errors.append("ledger accepted job differs")
    accounting_value, accounting_errors = accounting(args.accounting, args.job_id)
    errors.extend(accounting_errors)
    if (
        final.get("classification") != "NO RESULT"
        and (
            accounting_value.get("normalized_state") != "COMPLETED"
            or accounting_value.get("exit_code") != "0:0"
        )
    ):
        errors.append("scientific result requires COMPLETED with exit 0:0")
    current: dict[str, Any] = {}
    try:
        stage = Path(manifest["paths"]["stage"]).resolve(strict=True)
        current = build(stage)
        expected_stage = manifest["stage_inventory"]
        if (
            current.get("content_digest") != expected_stage["content_digest"]
            or current.get("record_count") != expected_stage["record_count"]
        ):
            errors.append("frozen stage changed after submission")
    except (KeyError, OSError, ValueError) as exc:
        errors.append(f"terminal stage inventory failed: {exc}")
    artifact_records = {}
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path == args.output:
            continue
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or path.is_symlink():
            errors.append(f"non-regular result artifact: {path.name}")
            continue
        if stat.S_IMODE(info.st_mode) & 0o077:
            errors.append(f"result artifact is not owner-only: {path.name}")
        artifact_records[path.name] = {
            "bytes": info.st_size,
            "mode": format(stat.S_IMODE(info.st_mode), "04o"),
            "sha256": file_sha256(path),
        }
    value = {
        "schema": "vq-phase-a-timing-terminal-validation/v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not errors else "failed",
        "classification": final.get("classification") if not errors else "NO RESULT",
        "job_id": args.job_id,
        "launch_manifest_sha256": args.manifest_sha256,
        "accounting": accounting_value,
        "stage_inventory_digest": current.get("content_digest"),
        "artifact_records": artifact_records,
        "errors": errors,
    }
    atomic_json(args.output, value)
    if errors:
        raise RuntimeError("terminal validation failed: " + "; ".join(errors))
    print(value["classification"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
