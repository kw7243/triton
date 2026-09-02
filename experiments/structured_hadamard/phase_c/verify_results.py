#!/usr/bin/env python3
"""Independently verify copied Phase C scientific and one-shot evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Mapping

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.structured_hadamard.phase_c import analysis, selector
from experiments.structured_hadamard.phase_c.schema import validate_records


class VerificationError(ValueError):
    """Copied Phase C evidence is incomplete or internally inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def verify(root: Path, *, policy_freeze: Path, phase_b_map: Path,
           phase_b_report: Path) -> dict[str, object]:
    scientific = root / "scientific-run"
    run_state = root / "run-state"
    results = _json(scientific / "results.json")
    measured = _jsonl(scientific / "measured-policies.jsonl")
    raw = _jsonl(scientific / "raw-timings.jsonl")
    assignments = _json(scientific / "policy-assignments.json")
    decision = _json(scientific / "literal-decision-c.json")
    freeze = _json(policy_freeze)

    selector.validate_freeze(freeze, map_path=phase_b_map, report_path=phase_b_report)
    _require(_sha256(policy_freeze) == results["policy_freeze"]["sha256"],
             "policy freeze digest differs")
    validate_records(measured, freeze["measurement_order"])
    _require(len(raw) == len(measured) * 5, "raw timing count differs")
    raw_keys = {(row["measurement_id"], row["metric"], row["repetition"]) for row in raw}
    _require(len(raw_keys) == len(raw), "raw timing keys are not unique")
    _require({row["metric"] for row in raw} == {"end_to_end_decode"},
             "raw timing metric differs")
    _require(assignments["policy_freeze_sha256"] == _sha256(policy_freeze),
             "assignment artifact freeze digest differs")
    _require(assignments["identity_to_measurement"] == freeze["identity_to_measurement"],
             "assignment artifact identity mapping differs")
    _require(assignments["measurements"] == freeze["measurements"],
             "measured assignment artifact differs from freeze")

    reproduced = analysis.literal_decision(measured, freeze)
    _require(decision == reproduced, "literal Decision C does not reproduce")
    _require(results["literal_decision_C"] == reproduced, "result Decision C differs")
    _require(results["phase_d_entered"] is False and results["second_scientific_run_entered"] is False,
             "forbidden dependent or second run was entered")
    _require(results["mixquant_perq"]["available"] is False,
             "unavailable MixQuant/PeRQ baseline was claimed")
    for name, expected in results["artifact_hashes"].items():
        _require(_sha256(scientific / name) == expected, f"artifact hash differs: {name}")

    expected_ledger = {
        "owner_attempts": 1, "salloc_attempts": 1, "srun_attempts": 1,
        "driver_attempts": 1, "terminal_events_fired": 1,
    }
    ledger = _json(run_state / "owner-ledger.json")
    terminal = _json(run_state / "remote-terminal.json")
    preflight = _json(run_state / "preflight-audit.json")
    _require(ledger == expected_ledger, "final one-shot ledger differs")
    _require(terminal["owner_exit"] == 0 and terminal["failure"] is None,
             "remote terminal was not successful")
    _require(terminal["ledger"] == expected_ledger, "terminal ledger differs")
    _require(preflight["status"] == "passed", "staged preflight did not pass")
    _require(preflight["policy"]["measurement_order"] == freeze["measurement_order"],
             "preflight did not bind frozen measurement order")
    _require(preflight["policy"]["source_rows_verified"] == 32 * len(measured),
             "preflight source-row hash count differs")

    return {
        "schema_version": "phase-c-result-verification-v1",
        "status": "passed",
        "root": str(root.resolve()),
        "source_commit": results["source"]["commit"],
        "policy_identities": len(freeze["identity_to_measurement"]),
        "unique_measurements": len(measured),
        "raw_timing_rows": len(raw),
        "decision": reproduced["decision"],
        "conclusion": reproduced["conclusion"],
        "mixquant_perq_available": False,
        "ledger": ledger,
        "input_hashes": {
            "policy-freeze.json": _sha256(policy_freeze),
            "results.json": _sha256(scientific / "results.json"),
            "measured-policies.jsonl": _sha256(scientific / "measured-policies.jsonl"),
            "raw-timings.jsonl": _sha256(scientific / "raw-timings.jsonl"),
            "literal-decision-c.json": _sha256(scientific / "literal-decision-c.json"),
            "owner-ledger.json": _sha256(run_state / "owner-ledger.json"),
            "remote-terminal.json": _sha256(run_state / "remote-terminal.json"),
        },
    }


def _write_exclusive(path: Path, value: Mapping[str, object]) -> None:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--policy-freeze", type=Path,
                        default=Path("data/rot-phasec-selector-r1/policy-freeze.json"))
    parser.add_argument("--phase-b-map", type=Path,
                        default=Path(selector.PHASE_B_MAP_REPO_PATH))
    parser.add_argument("--phase-b-report", type=Path,
                        default=Path(selector.PHASE_B_REPORT_REPO_PATH))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = verify(
        args.root.resolve(strict=True), policy_freeze=args.policy_freeze.resolve(strict=True),
        phase_b_map=args.phase_b_map.resolve(strict=True),
        phase_b_report=args.phase_b_report.resolve(strict=True),
    )
    if args.output:
        _write_exclusive(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
