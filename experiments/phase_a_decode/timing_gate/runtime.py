#!/usr/bin/env python3
"""Atomic runtime boundary and final-result records for the timing gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contract import FINAL_SCHEMA, RESULT_SCHEMA, SECONDARY_SIZES, classify, snapshot


FINAL_NAME = "final_result_manifest.json"
PAYLOAD_START_NAME = "payload_start_latch.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: Path, value: dict[str, Any], mode: int = 0o600) -> None:
    atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        mode,
    )


def exclusive_json(path: Path, value: dict[str, Any], mode: int = 0o400) -> None:
    """Create a durable one-time record without an overwrite path."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        parent = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def start(root: Path, digest: str, source: str, stage: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("invalid launch manifest SHA-256")
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id or re.fullmatch(r"[0-9]+", job_id) is None:
        raise RuntimeError("numeric SLURM_JOB_ID is required")
    allowed = {
        "attempt_ledger.jsonl",
        "environment.lock.txt",
        "environment_identity.json",
        "command.txt",
        "launch_manifest.json",
        "launch_manifest.sha256",
        "scheduler_options.json",
        "stage_inventory.json",
        "submission_request.json",
        f"slurm-{job_id}.out",
    }
    unexpected = sorted(path.name for path in root.iterdir() if path.name not in allowed)
    if unexpected:
        raise RuntimeError(f"unexpected pre-runtime artifacts: {unexpected}")
    payload = {
        "schema": "vq-phase-a-timing-runtime-start/v1",
        "recorded_at_utc": utc_now(),
        "job_id": job_id,
        "launch_manifest_sha256": digest,
        "source_repo": source,
        "staged_repo": stage,
        "result_root": str(root),
    }
    atomic_json(root / "runtime_start.json", payload)
    atomic_json(
        root / "execution_state.json",
        {
            "schema": "vq-phase-a-timing-execution-state/v1",
            "boundary": "runtime_started_before_validation",
            "recorded_at_utc": utc_now(),
            "launch_manifest_sha256": digest,
        },
    )


def payload_start(root: Path, digest: str, source: str, stage: str, route: str) -> None:
    """Consume the single scientific-payload budget immediately before benchmark exec."""
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("invalid launch manifest SHA-256")
    if route not in {"batch-primary", "interactive-fallback"}:
        raise ValueError("invalid launch route")
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id or re.fullmatch(r"[0-9]+", job_id) is None:
        raise RuntimeError("numeric SLURM_JOB_ID is required")
    validation = read_json(root / "environment_validation.json")
    if validation.get("status") != "passed":
        raise RuntimeError("environment validation did not pass")
    if validation.get("launch_manifest_sha256") != digest:
        raise RuntimeError("environment validation manifest binding mismatch")
    exclusive_json(
        root / PAYLOAD_START_NAME,
        {
            "schema": "vq-phase-a-timing-payload-start/v1",
            "recorded_at_utc": utc_now(),
            "job_id": job_id,
            "launch_route": route,
            "launch_manifest_sha256": digest,
            "source_repo": source,
            "staged_repo": stage,
            "result_root": str(root),
            "payload_start_ordinal": 1,
        },
    )


def _file_record(path: Path) -> dict[str, Any]:
    info = path.lstat()
    return {
        "bytes": info.st_size,
        "mode": format(stat.S_IMODE(info.st_mode), "04o"),
        "type": "file",
        "sha256": sha256(path),
    }


def validate_timing(root: Path, manifest_digest: str) -> tuple[str, list[str]]:
    errors: list[str] = []
    try:
        value = read_json(root / "timing_output.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return "NO RESULT", [f"timing output unavailable: {exc}"]
    if value.get("schema") != RESULT_SCHEMA:
        errors.append("timing output schema mismatch")
    if value.get("launch_manifest_sha256") != manifest_digest:
        errors.append("timing output manifest digest mismatch")
    if value.get("contract") != snapshot():
        errors.append("timing contract drift")
    rows = value.get("rows")
    if not isinstance(rows, list):
        return "NO RESULT", [*errors, "timing rows are missing"]
    expected_pairs = [(size, 32768) for size in SECONDARY_SIZES]
    if [(row.get("S"), row.get("Tkv")) for row in rows] != expected_pairs:
        errors.append("timing matrix differs from the two fixed production-like rows")
    for row in rows:
        variants = row.get("variants")
        if not isinstance(variants, dict) or set(variants) != {"J", "F", "H"}:
            errors.append("a timing row lacks exactly J/F/H")
            continue
        for variant, metrics in variants.items():
            if not isinstance(metrics, dict):
                errors.append(f"{variant} metrics are malformed")
                continue
            for key in ("p20_us", "p50_us", "p80_us", "dispersion", "outer_stability"):
                number = metrics.get(key)
                if (
                    isinstance(number, bool)
                    or not isinstance(number, (int, float))
                    or not math.isfinite(number)
                    or number < 0
                ):
                    errors.append(f"{variant} {key} is not finite nonnegative evidence")
    decision = classify(rows)
    if value.get("classification") != decision:
        errors.append("declared classification differs from fixed decision rule")
    if errors:
        return "NO RESULT", errors
    return decision, []


def finalize(
    root: Path,
    payload_exit: int,
    digest: str,
    source: str,
    stage: str,
) -> dict[str, Any]:
    classification, errors = validate_timing(root, digest)
    if payload_exit != 0:
        classification = "NO RESULT"
        errors.append(f"payload exit code {payload_exit}")
    atomic_json(
        root / "execution_state.json",
        {
            "schema": "vq-phase-a-timing-execution-state/v1",
            "boundary": "payload_terminal",
            "recorded_at_utc": utc_now(),
            "payload_exit_code": payload_exit,
            "classification": classification,
            "launch_manifest_sha256": digest,
        },
    )
    files = {
        path.name: _file_record(path)
        for path in sorted(root.iterdir(), key=lambda item: item.name)
        if path.is_file()
        and path.name != FINAL_NAME
        and not path.name.startswith("slurm-")
    }
    value = {
        "schema": FINAL_SCHEMA,
        "recorded_at_utc": utc_now(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "payload_exit_code": payload_exit,
        "classification": classification,
        "artifact_errors": errors,
        "launch_manifest_sha256": digest,
        "source_repo": source,
        "staged_repo": stage,
        "result_root": str(root),
        "payload_start_count": int((root / PAYLOAD_START_NAME).is_file()),
        "files": files,
    }
    atomic_json(root / FINAL_NAME, value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    begin = sub.add_parser("start")
    payload = sub.add_parser("payload-start")
    finish = sub.add_parser("finalize")
    for current in (begin, payload, finish):
        current.add_argument("--result", type=Path, required=True)
        current.add_argument("--manifest-sha256", required=True)
        current.add_argument("--source", required=True)
        current.add_argument("--stage", required=True)
    payload.add_argument("--route", required=True)
    finish.add_argument("--payload-exit-code", type=int, required=True)
    args = parser.parse_args()
    if args.command == "start":
        start(args.result, args.manifest_sha256, args.source, args.stage)
        return 0
    if args.command == "payload-start":
        payload_start(
            args.result,
            args.manifest_sha256,
            args.source,
            args.stage,
            args.route,
        )
        return 0
    value = finalize(
        args.result,
        args.payload_exit_code,
        args.manifest_sha256,
        args.source,
        args.stage,
    )
    print(value["classification"])
    return 0 if args.payload_exit_code == 0 and value["classification"] != "NO RESULT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
