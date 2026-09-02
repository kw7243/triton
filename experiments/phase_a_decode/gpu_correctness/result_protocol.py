#!/usr/bin/env python3
"""Atomic progress and fail-closed result classification for Phase A."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

try:
    from .protocol import (
        ABS_TOL,
        CONFIGS,
        FORBIDDEN_ARTIFACTS,
        INPUT_KINDS,
        REL_TOL,
        SECONDARY_SIZES,
        chunks_per_record,
        contract_snapshot,
    )
except ImportError:
    from protocol import (
        ABS_TOL,
        CONFIGS,
        FORBIDDEN_ARTIFACTS,
        INPUT_KINDS,
        REL_TOL,
        SECONDARY_SIZES,
        chunks_per_record,
        contract_snapshot,
    )


STATE_FILE = "execution_state.json"
FINAL_MANIFEST = "final_result_manifest.json"
NUMERICAL_CHECKS = {
    "finite",
    "J_bitwise_gather",
    "F_H_axis_bitwise_J",
    "F_H_vs_J_tolerance",
    "F_H_vs_quantized_oracle_tolerance",
    "F_H_vs_float32_oracle_tolerance",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, text: str, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, mode)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: dict[str, Any], *, mode: int = 0o600) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        mode=mode,
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def record_boundary(result_dir: Path, boundary: str, **details: Any) -> None:
    state = {
        "schema": "vq-phase-a-execution-state/v2",
        "recorded_at_utc": utc_now(),
        "boundary": boundary,
        **details,
    }
    atomic_write_json(result_dir / STATE_FILE, state)


def record_runtime_start(root: Path, manifest_sha256: str, source: str, stage: str) -> None:
    start = {
        "schema": "vq-phase-a-runtime-start/v2",
        "recorded_at_utc": utc_now(),
        "launch_manifest_sha256": manifest_sha256,
        "source_repo": source,
        "staged_repo": stage,
        "result_root": str(root),
    }
    atomic_write_json(root / "runtime_start.json", start)
    record_boundary(
        root,
        "payload_before_cuda",
        launch_manifest_sha256=manifest_sha256,
    )


def _expected_case_keys() -> set[tuple[int, str, str]]:
    return {
        (secondary_size, input_kind, config[2])
        for secondary_size in SECONDARY_SIZES
        for input_kind in INPUT_KINDS
        for config in CONFIGS
    }


def validate_passed_correctness(root: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    try:
        correctness = read_json(root / "correctness.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {}, [f"correctness artifact is unavailable or malformed: {exc}"]

    if correctness.get("schema") != "vq-phase-a-correctness/v2":
        errors.append("correctness schema mismatch")
    if correctness.get("status") != "passed":
        errors.append("correctness status is not passed")
    if correctness.get("scientific_classification") != "PASS":
        errors.append("scientific classification is not PASS")
    if correctness.get("mode") != "correctness-only":
        errors.append("mode is not correctness-only")
    for key in ("timing_executed", "tuning_executed", "cuda_graphs_executed"):
        if correctness.get(key) is not False:
            errors.append(f"{key} is not false")
    if correctness.get("tolerances") != {
        "absolute": ABS_TOL,
        "relative_frobenius": REL_TOL,
    }:
        errors.append("fixed tolerances changed")

    forbidden = [name for name in FORBIDDEN_ARTIFACTS if (root / name).exists()]
    if forbidden:
        errors.append("forbidden timing artifacts exist: " + ", ".join(forbidden))

    records = correctness.get("records")
    if not isinstance(records, list):
        errors.append("records are missing")
        return correctness, errors
    by_dtype: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if not isinstance(record, dict):
            errors.append("a correctness record is not an object")
            continue
        by_dtype.setdefault(str(record.get("dtype")), []).append(record)

    expected_dtypes = {"float16"}
    if correctness.get("bf16_supported") is True:
        expected_dtypes.add("bfloat16")
    if set(by_dtype) != expected_dtypes:
        errors.append(f"dtype records are {sorted(by_dtype)}, expected {sorted(expected_dtypes)}")
    expected_cases = _expected_case_keys()
    for dtype in expected_dtypes:
        observed_cases = {
            (record.get("S"), record.get("input"), record.get("config"))
            for record in by_dtype.get(dtype, [])
        }
        if observed_cases != expected_cases or len(by_dtype.get(dtype, [])) != len(expected_cases):
            errors.append(f"{dtype} does not contain exactly the eight required records")
        for record in by_dtype.get(dtype, []):
            try:
                expected_chunks = chunks_per_record(int(record["S"]), str(record["input"]))
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"invalid record identity: {exc}")
                continue
            if record.get("chunks") != expected_chunks:
                errors.append(
                    f"record {dtype}/S{record.get('S')}/{record.get('input')}/"
                    f"{record.get('config')} has {record.get('chunks')} chunks, "
                    f"expected {expected_chunks}"
                )
            checks = record.get("checks")
            required_checks = {
                "finite",
                "J_bitwise_gather",
                "F_H_axis_bitwise_J",
                "F_H_vs_J_tolerance",
                "F_H_vs_quantized_oracle_tolerance",
            }
            if not isinstance(checks, dict) or any(checks.get(name) is not True for name in required_checks):
                errors.append(
                    f"record {dtype}/S{record.get('S')}/{record.get('input')}/"
                    f"{record.get('config')} is missing a required passed check"
                )
            direct_expected = dtype == "float16"
            if not isinstance(checks, dict) or checks.get(
                "F_H_vs_float32_oracle_tolerance"
            ) is not direct_expected:
                errors.append(f"record {dtype} has incorrect direct-float32 threshold policy")
            metrics = record.get("metrics")
            required_metrics = {
                f"{variant}_vs_{reference}"
                for variant in ("F", "H")
                for reference in ("J", "oracle", "quantized_oracle")
            }
            if not isinstance(metrics, dict) or set(metrics) != required_metrics:
                errors.append(f"record {dtype} has incomplete comparison metrics")
                continue
            for name, metric in metrics.items():
                if not isinstance(metric, dict):
                    errors.append(f"record {dtype} metric {name} is not an object")
                    continue
                values = (metric.get("max_abs"), metric.get("relative_fro"))
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in values
                ):
                    errors.append(f"record {dtype} metric {name} is not finite numeric evidence")
                    continue
                threshold_applies = name.endswith("_vs_J") or name.endswith(
                    "_vs_quantized_oracle"
                ) or (dtype == "float16" and name.endswith("_vs_oracle"))
                if threshold_applies and (
                    metric["max_abs"] > ABS_TOL or metric["relative_fro"] > REL_TOL
                ):
                    errors.append(f"record {dtype} metric {name} exceeds its declared threshold")
    return correctness, errors


def mark_payload_complete(root: Path, manifest_sha256: str) -> None:
    correctness, errors = validate_passed_correctness(root)
    if errors:
        record_boundary(root, "after_artifact_writing", validation_errors=errors)
        raise RuntimeError("cannot complete payload: " + "; ".join(errors))
    metadata_path = root / "run_metadata.json"
    metadata = read_json(metadata_path)
    if metadata.get("launch_manifest_sha256") != manifest_sha256:
        record_boundary(root, "after_artifact_writing", error="manifest digest mismatch")
        raise RuntimeError("run metadata manifest digest mismatch")
    metadata.update(
        status="passed",
        correctness="PASS",
        run_complete=True,
        completed_at_utc=utc_now(),
    )
    atomic_write_json(metadata_path, metadata)
    record_boundary(
        root,
        "payload_complete",
        scientific_classification=correctness["scientific_classification"],
        launch_manifest_sha256=manifest_sha256,
    )


def _file_record(path: Path) -> dict[str, Any]:
    info = path.lstat()
    record: dict[str, Any] = {
        "bytes": info.st_size,
        "mode": format(stat.S_IMODE(info.st_mode), "04o"),
    }
    if path.is_symlink():
        target = os.readlink(path)
        record.update(
            type="symlink",
            target=target,
            sha256=hashlib.sha256(b"link\0" + os.fsencode(target)).hexdigest(),
        )
    elif path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        record.update(type="file", sha256=digest.hexdigest())
    elif path.is_dir():
        record.update(
            type="directory",
            sha256=hashlib.sha256(b"directory\0").hexdigest(),
        )
    else:
        record["type"] = "other"
    return record


def classify_result(root: Path, payload_exit_code: int) -> dict[str, Any]:
    state: dict[str, Any] = {}
    correctness: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    artifact_errors = []
    try:
        state = read_json(root / STATE_FILE)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        artifact_errors.append(f"execution state unavailable: {exc}")
    try:
        correctness = read_json(root / "correctness.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        artifact_errors.append(f"correctness unavailable: {exc}")
    try:
        metadata = read_json(root / "run_metadata.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        artifact_errors.append(f"run metadata unavailable: {exc}")

    correctness_status = correctness.get("status")
    declared_scientific = correctness.get("scientific_classification")
    failed_check = correctness.get("failed_check")
    numerical_failure = (
        correctness_status == "failed"
        and declared_scientific == "FAIL"
        and correctness.get("failure_kind") == "numerical_comparison"
        and failed_check in NUMERICAL_CHECKS
        and correctness.get("failure_phase") == "inside_comparison"
    )
    passed_artifact_errors: list[str] = []
    if correctness_status == "passed":
        _, passed_artifact_errors = validate_passed_correctness(root)
        artifact_errors.extend(passed_artifact_errors)
        required_metadata = {
            "schema": "vq-phase-a-run-metadata/v2",
            "status": "passed",
            "correctness": "PASS",
            "run_complete": True,
            "timing_executed": False,
            "tuning_executed": False,
            "cuda_graphs_executed": False,
        }
        for key, value in required_metadata.items():
            if metadata.get(key) != value:
                artifact_errors.append(f"run metadata {key} is not {value!r}")
        digest = metadata.get("launch_manifest_sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            artifact_errors.append("run metadata manifest digest is invalid")
        if state.get("launch_manifest_sha256") != digest:
            artifact_errors.append("execution state and run metadata manifest digests differ")

    if numerical_failure:
        classification = "FAIL"
        comparison = "FAIL"
        failure_phase = "inside_comparison"
    elif (
        payload_exit_code == 0
        and state.get("boundary") == "payload_complete"
        and correctness_status == "passed"
        and declared_scientific == "PASS"
        and not artifact_errors
    ):
        classification = "PASS"
        comparison = "PASS"
        failure_phase = None
    else:
        classification = "NO RESULT"
        comparison = (
            "PASS"
            if correctness_status == "passed" and not passed_artifact_errors
            else "NO RESULT"
        )
        boundary = state.get("boundary")
        failure_phase = (
            "after_artifact_writing"
            if correctness_status == "passed" or boundary in {"comparison_output_written", "after_artifact_writing"}
            else boundary or "before_payload_evidence"
        )

    return {
        "classification": classification,
        "scientific_comparison": comparison,
        "failure_phase": failure_phase,
        "failed_check": failed_check if numerical_failure else None,
        "artifact_errors": artifact_errors,
    }


def finalize_result(
    root: Path,
    payload_exit_code: int,
    source: str,
    stage: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    result = classify_result(root, payload_exit_code)
    files = {
        path.name: _file_record(path)
        for path in sorted(root.iterdir(), key=lambda item: item.name)
        if path.name != FINAL_MANIFEST
    }
    manifest = {
        "schema": "vq-phase-a-gpu-correctness-r1/v2",
        "recorded_at_utc": utc_now(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "payload_exit_code": payload_exit_code,
        "source_repo": source,
        "staged_repo": stage,
        "result_root": str(root),
        "launch_manifest_sha256": manifest_sha256,
        "correctness_contract": contract_snapshot(bf16_supported=None),
        "timing_or_decision_artifacts_present": sorted(
            name for name in FORBIDDEN_ARTIFACTS if (root / name).exists()
        ),
        "files": files,
        **result,
    }
    atomic_write_json(root / FINAL_MANIFEST, manifest)
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    start = subparsers.add_parser("start")
    start.add_argument("--result", type=Path, required=True)
    start.add_argument("--manifest-sha256", required=True)
    start.add_argument("--source", required=True)
    start.add_argument("--stage", required=True)
    complete = subparsers.add_parser("complete")
    complete.add_argument("--result", type=Path, required=True)
    complete.add_argument("--manifest-sha256", required=True)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--result", type=Path, required=True)
    finalize.add_argument("--payload-exit-code", type=int, required=True)
    finalize.add_argument("--source", required=True)
    finalize.add_argument("--stage", required=True)
    finalize.add_argument("--manifest-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "start":
        record_runtime_start(
            args.result,
            args.manifest_sha256,
            args.source,
            args.stage,
        )
    elif args.command == "complete":
        mark_payload_complete(args.result, args.manifest_sha256)
    else:
        manifest = finalize_result(
            args.result,
            args.payload_exit_code,
            args.source,
            args.stage,
            args.manifest_sha256,
        )
        print(manifest["classification"])
        if args.payload_exit_code == 0 and manifest["classification"] != "PASS":
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
