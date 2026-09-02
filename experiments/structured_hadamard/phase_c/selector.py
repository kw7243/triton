"""Dependency-free deterministic Phase C policy construction and validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

from experiments.structured_hadamard.phase_b.schema import validate_rows


SCHEMA_VERSION = "phase-c-policy-freeze-v1"
TRANSFORM_ORDER = ("I", "H32", "H128", "Hfull")
TARGET_BUDGET_PERCENT = (0, 25, 50, 75, 100)
PHASE_B_PROVENANCE_TIP = "f7cd759307ffdb534581587520b42e95a309498c"
PHASE_B_MAP_SHA256 = "b5461aa9af17a85936f55c57c9c55d754a774236313c41363b62b6f849d8ef62"
PHASE_B_REPORT_SHA256 = "896d74da1e00d56bb260bd39a56d0cc42e66b3554916a67c808b8b9c44d53f87"
PHASE_B_MAP_REPO_PATH = "data/rot-phaseb-map-r1/artifacts/run-1662528/scientific-run/map-rows.jsonl"
PHASE_B_REPORT_REPO_PATH = "data/rot-phaseb-map-r1/report.md"


class SelectorError(ValueError):
    """Frozen Phase B data cannot produce the exact Phase C contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_rows(path: Path) -> tuple[list[dict[str, object]], dict[tuple[int, str], dict[str, object]]]:
    if sha256_file(path) != PHASE_B_MAP_SHA256:
        raise SelectorError("accepted Phase B map digest differs")
    raw_lines = path.read_bytes().splitlines()
    if len(raw_lines) != 128 or any(not line for line in raw_lines):
        raise SelectorError("accepted Phase B map must contain exactly 128 nonempty JSONL rows")
    rows = [json.loads(line) for line in raw_lines]
    validate_rows(rows)
    source = {}
    for line_number, (line, row) in enumerate(zip(raw_lines, rows), start=1):
        key = (int(row["layer"]), str(row["transform"]))
        source[key] = {
            "line": line_number,
            "sha256": hashlib.sha256(line).hexdigest(),
            "row": row,
        }
    return rows, source


def _cost(row: Mapping[str, object]) -> float:
    return float(row["timing"]["transform"]["median_ms"])


def _error(row: Mapping[str, object]) -> float:
    return float(row["quality"]["full"]["normalized_output_error"])


def _assignment_record(assignments: Mapping[int, str],
                       source: Mapping[tuple[int, str], Mapping[str, object]]) -> list[dict[str, object]]:
    records = []
    for layer in range(32):
        transform = assignments[layer]
        item = source[layer, transform]
        row = item["row"]
        records.append({
            "layer": layer,
            "site": f"model.layers.{layer}.mlp.down_proj",
            "transform": transform,
            "phase_b_source_line": item["line"],
            "phase_b_source_row_sha256": item["sha256"],
            "predicted_local_nmse": _error(row),
            "measured_transform_median_ms": _cost(row),
        })
    return records


def _summarize(assignments: Mapping[int, str],
               source: Mapping[tuple[int, str], Mapping[str, object]],
               full_cost: float) -> dict[str, object]:
    cost = math.fsum(_cost(source[layer, assignments[layer]]["row"]) for layer in range(32))
    loss = math.fsum(
        _error(source[layer, assignments[layer]]["row"]) - _error(source[layer, "Hfull"]["row"])
        for layer in range(32)
    )
    encoded = {str(layer): assignments[layer] for layer in range(32)}
    return {
        "assignment_sha256": canonical_sha256(encoded),
        "assignments": encoded,
        "source_rows": _assignment_record(assignments, source),
        "realized_predicted_transform_cost_ms": cost,
        "realized_predicted_transform_cost_percent": 100.0 * cost / full_cost,
        "predicted_aggregate_local_nmse_loss_vs_hfull": loss,
    }


def _downgrade_trace(source: Mapping[tuple[int, str], Mapping[str, object]]) -> tuple[float, list[dict[str, object]]]:
    assignments = {layer: "Hfull" for layer in range(32)}
    full_cost = math.fsum(_cost(source[layer, "Hfull"]["row"]) for layer in range(32))
    trace = [{"step": 0, **_summarize(assignments, source, full_cost), "move": None}]
    while any(transform != "I" for transform in assignments.values()):
        candidates = []
        for layer in range(32):
            current = assignments[layer]
            rank = TRANSFORM_ORDER.index(current)
            if rank == 0:
                continue
            cheaper = TRANSFORM_ORDER[rank - 1]
            current_row = source[layer, current]["row"]
            cheaper_row = source[layer, cheaper]["row"]
            latency_saved = _cost(current_row) - _cost(cheaper_row)
            if not math.isfinite(latency_saved) or latency_saved <= 0.0:
                raise SelectorError(
                    f"adjacent downgrade lacks strictly positive latency saving: layer={layer} {current}->{cheaper}"
                )
            quality_loss = _error(cheaper_row) - _error(current_row)
            rho = quality_loss / latency_saved
            if not math.isfinite(rho):
                raise SelectorError("downgrade rho must be finite")
            candidates.append((rho, layer, rank - 1, current, cheaper, quality_loss, latency_saved))
        if not candidates:
            raise SelectorError("downgrade trace stopped before reaching identity")
        rho, layer, _target_rank, current, cheaper, quality_loss, latency_saved = min(candidates)
        assignments[layer] = cheaper
        trace.append({
            "step": len(trace),
            **_summarize(assignments, source, full_cost),
            "move": {
                "layer": layer,
                "site": f"model.layers.{layer}.mlp.down_proj",
                "from": current,
                "to": cheaper,
                "predicted_quality_loss": quality_loss,
                "measured_latency_saved_ms": latency_saved,
                "rho": rho,
                "tie_break_key": [rho, layer, _target_rank],
            },
        })
    if len(trace) != 97 or trace[-1]["realized_predicted_transform_cost_ms"] != 0.0:
        raise SelectorError("complete trace must contain 96 adjacent downgrades and end at zero cost")
    return full_cost, trace


def _budget_policy(percent: int, *, full_cost: float, trace: Sequence[Mapping[str, object]]) -> dict[str, object]:
    target = full_cost * percent / 100.0
    matches = [record for record in trace
               if float(record["realized_predicted_transform_cost_ms"]) <= target + 1.0e-15]
    if not matches:
        raise SelectorError(f"budget {percent}% is infeasible")
    selected = min(matches, key=lambda record: int(record["step"]))
    if int(selected["step"]) > 0:
        previous = trace[int(selected["step"]) - 1]
        if float(previous["realized_predicted_transform_cost_ms"]) <= target + 1.0e-15:
            raise SelectorError("budget did not select the earliest feasible trace prefix")
    return {
        "policy_id": f"adaptive-budget-{percent:03d}",
        "kind": "adaptive",
        "target_budget_percent": percent,
        "target_predicted_transform_cost_ms": target,
        "trace_prefix_steps": int(selected["step"]),
        "budget_rule": "earliest downgrade-trace prefix with realized predicted transform cost <= target",
        **{key: selected[key] for key in (
            "assignment_sha256", "assignments", "source_rows",
            "realized_predicted_transform_cost_ms", "realized_predicted_transform_cost_percent",
            "predicted_aggregate_local_nmse_loss_vs_hfull",
        )},
    }


def freeze_policies(map_path: Path, report_path: Path) -> dict[str, object]:
    if sha256_file(report_path) != PHASE_B_REPORT_SHA256:
        raise SelectorError("accepted Phase B report digest differs")
    _rows, source = _read_rows(map_path)
    full_cost, trace = _downgrade_trace(source)
    adaptive = [_budget_policy(percent, full_cost=full_cost, trace=trace)
                for percent in TARGET_BUDGET_PERCENT]
    fixed = []
    for transform in TRANSFORM_ORDER:
        assignments = {layer: transform for layer in range(32)}
        fixed.append({
            "policy_id": f"fixed-{transform.lower()}",
            "kind": "fixed_baseline",
            "global_transform": transform,
            **_summarize(assignments, source, full_cost),
        })

    identities = [*adaptive, *fixed]
    measurements: dict[str, dict[str, object]] = {}
    identity_to_measurement = {}
    measurement_order = []
    by_assignment = {}
    for policy in identities:
        assignment_sha = str(policy["assignment_sha256"])
        if assignment_sha not in by_assignment:
            measurement_id = f"measurement-{len(measurement_order):02d}-{assignment_sha[:12]}"
            by_assignment[assignment_sha] = measurement_id
            measurement_order.append(measurement_id)
            measurements[measurement_id] = {
                "measurement_id": measurement_id,
                "assignment_sha256": assignment_sha,
                "assignments": policy["assignments"],
                "source_rows": policy["source_rows"],
                "realized_predicted_transform_cost_ms": policy["realized_predicted_transform_cost_ms"],
                "realized_predicted_transform_cost_percent": policy["realized_predicted_transform_cost_percent"],
                "predicted_aggregate_local_nmse_loss_vs_hfull":
                    policy["predicted_aggregate_local_nmse_loss_vs_hfull"],
                "policy_aliases": [],
            }
        measurement_id = by_assignment[assignment_sha]
        identity_to_measurement[policy["policy_id"]] = measurement_id
        measurements[measurement_id]["policy_aliases"].append(policy["policy_id"])

    freeze = {
        "schema_version": SCHEMA_VERSION,
        "phase_b": {
            "provenance_tip": PHASE_B_PROVENANCE_TIP,
            "map_path": PHASE_B_MAP_REPO_PATH,
            "map_sha256": PHASE_B_MAP_SHA256,
            "report_path": PHASE_B_REPORT_REPO_PATH,
            "report_sha256": PHASE_B_REPORT_SHA256,
            "source_row_hash_definition": "SHA-256 of the exact UTF-8 JSONL row bytes excluding the LF terminator",
        },
        "selector": {
            "direction": "downgrade from Hfull",
            "transform_order_cheapest_to_most_expensive": list(TRANSFORM_ORDER),
            "atomic_moves": "adjacent only: Hfull->H128, H128->H32, H32->I",
            "quality_loss": "Phase B full-calibration local NMSE(to) - local NMSE(from)",
            "latency_saved": "Phase B measured transform median_ms(from) - median_ms(to)",
            "rho": "predicted quality loss / measured latency saved",
            "feasible_move": "the layer currently uses the move's from-transform and latency_saved is strictly positive",
            "selection": "repeatedly apply the feasible move with the lowest tie-break key",
            "tie_break": "ascending (rho, layer index, target transform rank in I<H32<H128<Hfull)",
            "budget_accounting": "sum of the 32 assigned Phase B measured transform median_ms values",
            "budget_feasibility": "earliest trace prefix at or below target; overshoot is allowed and recorded",
            "monotonicity": "lower budgets are longer prefixes of one trace; no layer is upgraded",
            "full_cost_ms": full_cost,
            "trace_steps": len(trace) - 1,
            "trace_sha256": canonical_sha256(trace),
        },
        "decision_contract": {
            "quality_good_ppl_delta_max": 0.2,
            "C1_speedup_percent_min": 3.0,
            "C2_speedup_percent_min_inclusive": 1.0,
            "C2_speedup_percent_max_exclusive": 3.0,
            "material_latency_improvement_percent": 1.0,
            "material_ppl_improvement": 0.05,
            "no_worse_latency_tolerance_percent": 0.5,
            "no_worse_ppl_tolerance": 0.02,
            "multiple_adaptive_budgets": 2,
        },
        "target_budgets_percent": list(TARGET_BUDGET_PERCENT),
        "adaptive_policies": adaptive,
        "fixed_baselines": fixed,
        "identity_to_measurement": identity_to_measurement,
        "measurement_order": measurement_order,
        "measurements": measurements,
        "deduplication": {
            "rule": "exact assignment_sha256 equality",
            "identity_count": len(identities),
            "unique_measurement_count": len(measurements),
            "endpoint_identity_mapping_preserved": True,
        },
        "mixquant_perq": {
            "available": False,
            "reason": "No pinned, already-reproducible comparable MixQuant/PeRQ implementation is present in the accepted repository or dependency set; reproduction is outside this lane.",
        },
        "frozen_before_phase_c_outcomes": True,
    }
    validate_freeze(freeze)
    return freeze


def validate_freeze(value: Mapping[str, object], *, map_path: Path | None = None,
                    report_path: Path | None = None) -> None:
    if value.get("schema_version") != SCHEMA_VERSION:
        raise SelectorError("policy freeze schema differs")
    adaptive = value.get("adaptive_policies")
    fixed = value.get("fixed_baselines")
    if not isinstance(adaptive, list) or [row.get("target_budget_percent") for row in adaptive] != list(TARGET_BUDGET_PERCENT):
        raise SelectorError("five adaptive budget policies differ")
    if not isinstance(fixed, list) or [row.get("global_transform") for row in fixed] != list(TRANSFORM_ORDER):
        raise SelectorError("four fixed global baselines differ")
    ranks = {transform: index for index, transform in enumerate(TRANSFORM_ORDER)}
    previous = None
    previous_cost = -math.inf
    for policy in adaptive:
        assignments = policy["assignments"]
        if set(assignments) != {str(layer) for layer in range(32)}:
            raise SelectorError("adaptive assignment does not cover exactly 32 layers")
        cost = float(policy["realized_predicted_transform_cost_ms"])
        if cost + 1.0e-12 < previous_cost:
            raise SelectorError("adaptive realized cost is not monotone with budget")
        if previous is not None and any(ranks[previous[str(layer)]] > ranks[assignments[str(layer)]]
                                        for layer in range(32)):
            raise SelectorError("higher budget downgraded a layer relative to a lower budget")
        previous, previous_cost = assignments, cost
    identities = [*adaptive, *fixed]
    mapping = value.get("identity_to_measurement")
    measurements = value.get("measurements")
    if not isinstance(mapping, dict) or not isinstance(measurements, dict):
        raise SelectorError("measurement deduplication tables are malformed")
    for policy in identities:
        measurement = measurements.get(mapping.get(policy["policy_id"]))
        if not isinstance(measurement, dict) or measurement["assignment_sha256"] != policy["assignment_sha256"]:
            raise SelectorError("policy identity does not map to its exact assignment")
        if len(policy["source_rows"]) != 32:
            raise SelectorError("policy lacks 32 Phase B source-row hashes")
    if value.get("frozen_before_phase_c_outcomes") is not True:
        raise SelectorError("policy freeze timing attestation differs")
    if map_path is not None and report_path is not None:
        expected = freeze_policies(map_path, report_path)
        if value != expected:
            raise SelectorError("policy freeze does not reproduce byte-for-object from accepted Phase B inputs")


def _write_exclusive(path: Path, value: object) -> None:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-b-map", type=Path, required=True)
    parser.add_argument("--phase-b-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = freeze_policies(args.phase_b_map.resolve(strict=True), args.phase_b_report.resolve(strict=True))
    _write_exclusive(args.output, value)
    print(json.dumps({
        "output": str(args.output), "sha256": sha256_file(args.output),
        "unique_measurements": value["deduplication"]["unique_measurement_count"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
