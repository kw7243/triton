"""Phase C Pareto table, figure, and predeclared literal Decision C logic."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


class AnalysisError(ValueError):
    """Measured policies do not satisfy the frozen analysis contract."""


def _point(policy_id: str, measurements: Mapping[str, Mapping[str, object]],
           identity_to_measurement: Mapping[str, str]) -> dict[str, object]:
    measurement_id = identity_to_measurement[policy_id]
    record = measurements[measurement_id]
    return {
        "policy_id": policy_id,
        "measurement_id": measurement_id,
        "perplexity": float(record["quality"]["perplexity"]),
        "decode_median_ms": float(record["decode"]["median_ms"]),
        "decode_median_ms_per_output_token": float(record["decode"]["median_ms_per_output_token"]),
        "decode_median_tokens_per_second": float(record["decode"]["median_tokens_per_second"]),
        "realized_predicted_transform_cost_ms":
            float(record["realized_predicted_transform_cost_ms"]),
        "realized_predicted_transform_cost_percent":
            float(record["realized_predicted_transform_cost_percent"]),
    }


def _speedup_percent(point: Mapping[str, object], reference: Mapping[str, object]) -> float:
    latency = float(point["decode_median_ms"])
    baseline = float(reference["decode_median_ms"])
    if latency <= 0 or baseline <= 0:
        raise AnalysisError("decode medians must be positive")
    return (baseline / latency - 1.0) * 100.0


def _materially_dominates(lhs: Mapping[str, object], rhs: Mapping[str, object],
                          contract: Mapping[str, object]) -> bool:
    lhs_ppl, rhs_ppl = float(lhs["perplexity"]), float(rhs["perplexity"])
    lhs_ms, rhs_ms = float(lhs["decode_median_ms"]), float(rhs["decode_median_ms"])
    latency_tolerance = float(contract["no_worse_latency_tolerance_percent"])
    ppl_tolerance = float(contract["no_worse_ppl_tolerance"])
    no_worse = (
        lhs_ppl <= rhs_ppl + ppl_tolerance
        and lhs_ms <= rhs_ms * (1.0 + latency_tolerance / 100.0)
    )
    materially_better = (
        lhs_ppl <= rhs_ppl - float(contract["material_ppl_improvement"])
        or _speedup_percent(lhs, rhs) >= float(contract["material_latency_improvement_percent"])
    )
    return no_worse and materially_better


def literal_decision(measured: Sequence[Mapping[str, object]], freeze: Mapping[str, object]) -> dict[str, object]:
    by_measurement = {str(row["measurement_id"]): row for row in measured}
    expected = set(freeze["measurement_order"])
    if set(by_measurement) != expected or len(by_measurement) != len(measured):
        raise AnalysisError("measured policies do not match the unique frozen measurement set")
    mapping = freeze["identity_to_measurement"]
    points = {policy_id: _point(policy_id, by_measurement, mapping)
              for policy_id in mapping}
    hfull = points["fixed-hfull"]
    contract = freeze["decision_contract"]
    adaptive_ids = [f"adaptive-budget-{budget:03d}" for budget in (25, 50, 75)]
    fixed_block_ids = ("fixed-h32", "fixed-h128")
    fixed_ids = ("fixed-i", "fixed-h32", "fixed-h128", "fixed-hfull")

    comparisons = []
    frontier_winners = set()
    dominated_fixed_blocks = set()
    for adaptive_id in adaptive_ids:
        adaptive = points[adaptive_id]
        dominated_by_fixed = any(
            _materially_dominates(points[fixed_id], adaptive, contract) for fixed_id in fixed_ids
        )
        dominates = [fixed_id for fixed_id in fixed_block_ids
                     if _materially_dominates(adaptive, points[fixed_id], contract)]
        if dominates and not dominated_by_fixed:
            frontier_winners.add(adaptive_id)
            dominated_fixed_blocks.update(dominates)
        comparisons.append({
            "policy_id": adaptive_id,
            "ppl_delta_vs_hfull": float(adaptive["perplexity"]) - float(hfull["perplexity"]),
            "speedup_percent_vs_hfull": _speedup_percent(adaptive, hfull),
            "materially_dominates_fixed_blocks": dominates,
            "materially_dominated_by_any_fixed": dominated_by_fixed,
            "beats_fixed_frontier": bool(dominates and not dominated_by_fixed),
        })

    selector_beats_fixed = bool(frontier_winners)
    multi_budget_dominance = (
        len(frontier_winners) >= int(contract["multiple_adaptive_budgets"])
        and dominated_fixed_blocks == set(fixed_block_ids)
    )
    quality_good = [row for row in comparisons
                    if row["ppl_delta_vs_hfull"] <= float(contract["quality_good_ppl_delta_max"])]
    best_quality_good_speedup = max(
        (float(row["speedup_percent_vs_hfull"]) for row in quality_good), default=-math.inf,
    )
    strong_threshold = (
        best_quality_good_speedup >= float(contract["C1_speedup_percent_min"])
        or multi_budget_dominance
    )
    moderate_threshold = (
        best_quality_good_speedup >= float(contract["C2_speedup_percent_min_inclusive"])
        and best_quality_good_speedup < float(contract["C2_speedup_percent_max_exclusive"])
    )
    maximum_speedup = max(float(row["speedup_percent_vs_hfull"]) for row in comparisons)

    if not selector_beats_fixed:
        code = "C4"
        conclusion = "C4: adaptive allocation does not beat the fixed-transform frontier; stop the per-layer allocation idea"
        next_step = "Stop; do not add selector search complexity."
    elif strong_threshold:
        code = "C1"
        conclusion = "C1: adaptive allocation gives a strong quality-latency or multi-budget Pareto win"
        next_step = "The authoritative plan owns routing to Phase D; Phase D was not entered here."
    elif moderate_threshold:
        code = "C2"
        conclusion = "C2: adaptive allocation preserves quality with only a 1-3% end-to-end speedup"
        next_step = (
            "Single profiling question only: does sequential transform-path kernel-launch or global-memory overhead prevent "
            "the Phase B predicted transform-cost saving from appearing in end-to-end decode latency?"
        )
    elif maximum_speedup > 0.0:
        code = "C3"
        conclusion = "C3: adaptive allocation improves speed, but its quality loss is too large"
        next_step = (
            "One dependent recovery only: a MixQuant/PeRQ-style offline mass-balancing permutation before smaller block "
            "rotations; it was not implemented or run here."
        )
    else:
        code = "C4"
        conclusion = "C4: adaptive allocation has no end-to-end win over the fixed-transform frontier; stop"
        next_step = "Stop; do not add selector search complexity."

    return {
        "schema_version": "phase-c-literal-decision-v1",
        "contract": dict(contract),
        "endpoint_deduplication": {
            "adaptive_budget_000_equals_fixed_i": mapping["adaptive-budget-000"] == mapping["fixed-i"],
            "adaptive_budget_100_equals_fixed_hfull":
                mapping["adaptive-budget-100"] == mapping["fixed-hfull"],
        },
        "adaptive_comparisons": comparisons,
        "fixed_frontier_winning_adaptive_policies": sorted(frontier_winners),
        "fixed_block_baselines_dominated": sorted(dominated_fixed_blocks),
        "selector_beats_fixed_frontier": selector_beats_fixed,
        "multi_budget_fixed_baseline_dominance": multi_budget_dominance,
        "best_quality_good_speedup_percent_vs_hfull":
            None if best_quality_good_speedup == -math.inf else best_quality_good_speedup,
        "maximum_interior_adaptive_speedup_percent_vs_hfull": maximum_speedup,
        "decision": code,
        "conclusion": conclusion,
        "next_step": next_step,
        "phase_d_entered": False,
        "c_kernel_entered": False,
        "accuracy_recovery_entered": False,
        "second_scientific_run_entered": False,
    }


def identity_rows(measured: Sequence[Mapping[str, object]], freeze: Mapping[str, object]) -> list[dict[str, object]]:
    by_measurement = {str(row["measurement_id"]): row for row in measured}
    adaptive_meta = {row["policy_id"]: row for row in freeze["adaptive_policies"]}
    fixed_meta = {row["policy_id"]: row for row in freeze["fixed_baselines"]}
    rows = []
    identities = [*(f"adaptive-budget-{budget:03d}" for budget in (0, 25, 50, 75, 100)),
                  "fixed-i", "fixed-h32", "fixed-h128", "fixed-hfull"]
    for policy_id in identities:
        point = _point(policy_id, by_measurement, freeze["identity_to_measurement"])
        meta = adaptive_meta.get(policy_id, fixed_meta.get(policy_id))
        point.update({
            "kind": meta["kind"],
            "target_budget_percent": meta.get("target_budget_percent"),
            "global_transform": meta.get("global_transform"),
            "assignment_sha256": meta["assignment_sha256"],
        })
        rows.append(point)
    return rows


def write_table(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "policy_id", "kind", "measurement_id", "target_budget_percent", "global_transform",
            "realized_predicted_transform_cost_ms", "realized_predicted_transform_cost_percent",
            "perplexity", "decode_median_ms", "decode_median_ms_per_output_token",
            "decode_median_tokens_per_second", "assignment_sha256",
        ))
        for row in rows:
            writer.writerow(tuple(row.get(key) for key in (
                "policy_id", "kind", "measurement_id", "target_budget_percent", "global_transform",
                "realized_predicted_transform_cost_ms", "realized_predicted_transform_cost_percent",
                "perplexity", "decode_median_ms", "decode_median_ms_per_output_token",
                "decode_median_tokens_per_second", "assignment_sha256",
            )))


def write_figure(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    width, height = 1000, 680
    adaptive = [row for row in rows if row["kind"] == "adaptive"]
    fixed = [row for row in rows if row["kind"] == "fixed_baseline"]
    x_values = [float(row["decode_median_ms_per_output_token"]) for row in rows]
    y_values = [float(row["perplexity"]) for row in rows]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_pad, y_pad = max((x_max - x_min) * 0.08, 1e-9), max((y_max - y_min) * 0.08, 1e-9)
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad
    left, top, plot_w, plot_h = 90, 55, 855, 525

    def xy(row):
        x = left + (float(row["decode_median_ms_per_output_token"]) - x_min) / (x_max - x_min) * plot_w
        y = top + (1.0 - (float(row["perplexity"]) - y_min) / (y_max - y_min)) * plot_h
        return x, y

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#111827}.grid{stroke:#e5e7eb}.axis{stroke:#374151;stroke-width:1.5}</style>',
        '<text x="500" y="28" text-anchor="middle" font-size="20">Phase C five-budget quality-latency Pareto curve</text>',
    ]
    for tick in range(6):
        fraction = tick / 5
        x, y = left + fraction * plot_w, top + (1 - fraction) * plot_h
        parts.extend((
            f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}"/>',
            f'<text x="{x:.2f}" y="{top + plot_h + 24}" text-anchor="middle" font-size="12">{x_min + fraction * (x_max - x_min):.3g}</text>',
            f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}"/>',
            f'<text x="{left - 9}" y="{y + 4:.2f}" text-anchor="end" font-size="12">{y_min + fraction * (y_max - y_min):.4g}</text>',
        ))
    parts.extend((
        f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>',
        f'<text x="{left + plot_w / 2}" y="642" text-anchor="middle" font-size="15">End-to-end decode median (ms/output token; lower is better)</text>',
        f'<text transform="translate(23 {top + plot_h / 2}) rotate(-90)" text-anchor="middle" font-size="15">WikiText-2 perplexity (lower is better)</text>',
    ))
    ordered = sorted(adaptive, key=lambda row: int(row["target_budget_percent"]))
    polyline = " ".join(f"{xy(row)[0]:.2f},{xy(row)[1]:.2f}" for row in ordered)
    parts.append(f'<polyline points="{polyline}" fill="none" stroke="#2563eb" stroke-width="2.5"/>')
    for row in ordered:
        x, y = xy(row)
        label = f'{int(row["target_budget_percent"])}%'
        parts.extend((
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="#2563eb"><title>{row["policy_id"]}</title></circle>',
            f'<text x="{x + 8:.2f}" y="{y - 8:.2f}" font-size="12" fill="#1d4ed8">{label}</text>',
        ))
    for row in fixed:
        x, y = xy(row)
        label = str(row["global_transform"])
        parts.extend((
            f'<rect x="{x - 5:.2f}" y="{y - 5:.2f}" width="10" height="10" fill="#dc2626"><title>{row["policy_id"]}</title></rect>',
            f'<text x="{x + 8:.2f}" y="{y + 16:.2f}" font-size="12">{label}</text>',
        ))
    parts.append('</svg>')
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
