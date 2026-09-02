"""Frozen selection, stability, proxy-validation, and literal Decision B logic."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics
from typing import Iterable, Mapping, Sequence


PRIMARY_PROXY = "local_nmse_h32_minus_hfull"
STRONGER_PROXY = "short_sequence_end_to_end_nll_h32_minus_hfull"
CHEAPER_TRANSFORM = "H32"
SUBSET_NAMES = ("calibration_sequences_0_7", "calibration_sequences_8_15")
STABILITY_STATISTIC = "Spearman rank correlation of the 32 site sensitivity scores"
STABILITY_THRESHOLD = 0.5
QUALITY_HETEROGENEITY_RELATIVE_SPREAD_THRESHOLD = 0.5
REALIZED_LATENCY_GAP_THRESHOLD_PERCENT = 5.0


class AnalysisError(ValueError):
    """Rows are incomplete or violate the predeclared Phase B analysis."""


def _rank(values: Mapping[int, float]) -> dict[int, float]:
    if not values or any(not math.isfinite(value) for value in values.values()):
        raise AnalysisError("rank values must be nonempty and finite")
    ordered = sorted(values, key=lambda layer: (values[layer], layer))
    result: dict[int, float] = {}
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
            end += 1
        average = (cursor + 1 + end) / 2.0
        for index in range(cursor, end):
            result[ordered[index]] = average
        cursor = end
    return result


def spearman_rho(lhs: Mapping[int, float], rhs: Mapping[int, float]) -> float:
    if set(lhs) != set(rhs) or len(lhs) < 2:
        raise AnalysisError("Spearman inputs must have the same two-or-more site keys")
    lhs_rank, rhs_rank = _rank(lhs), _rank(rhs)
    left = [lhs_rank[key] for key in sorted(lhs)]
    right = [rhs_rank[key] for key in sorted(rhs)]
    left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left) * sum((b - right_mean) ** 2 for b in right)
    )
    return numerator / denominator if denominator else 0.0


def _rows_by_layer_transform(rows: Sequence[Mapping[str, object]]) -> dict[tuple[int, str], Mapping[str, object]]:
    indexed = {}
    for row in rows:
        key = (int(row["layer"]), str(row["transform"]))
        if key in indexed:
            raise AnalysisError(f"duplicate map tuple: {key}")
        indexed[key] = row
    expected = {(layer, transform) for layer in range(32) for transform in ("I", "H32", "H128", "Hfull")}
    if set(indexed) != expected:
        raise AnalysisError("map must contain exactly 32 layers times four transforms")
    return indexed


def primary_scores(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[int, float]]:
    indexed = _rows_by_layer_transform(rows)
    by_subset: dict[str, dict[int, float]] = {name: {} for name in SUBSET_NAMES}
    for layer in range(32):
        for name in SUBSET_NAMES:
            h32 = indexed[layer, "H32"]["quality"]["subsets"][name]["normalized_output_error"]
            hfull = indexed[layer, "Hfull"]["quality"]["subsets"][name]["normalized_output_error"]
            by_subset[name][layer] = float(h32) - float(hfull)
    by_subset["mean"] = {
        layer: statistics.fmean(by_subset[name][layer] for name in SUBSET_NAMES)
        for layer in range(32)
    }
    return by_subset


def freeze_selection(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    scores = primary_scores(rows)
    ordered = sorted(scores["mean"], key=lambda layer: (-scores["mean"][layer], layer))
    rho = spearman_rho(scores[SUBSET_NAMES[0]], scores[SUBSET_NAMES[1]])
    return {
        "schema_version": "phase-b-selection-freeze-v1",
        "proxy": PRIMARY_PROXY,
        "definition": "NMSE(H32) - NMSE(Hfull) at each FFN down_proj input",
        "cheaper_transform": CHEAPER_TRANSFORM,
        "subset_definitions": {
            SUBSET_NAMES[0]: {"sequence_indices": list(range(0, 8)), "rows": 4096},
            SUBSET_NAMES[1]: {"sequence_indices": list(range(8, 16)), "rows": 4096},
        },
        "stability": {
            "statistic": STABILITY_STATISTIC, "spearman_rho": rho,
            "stable_threshold": STABILITY_THRESHOLD, "stable": rho >= STABILITY_THRESHOLD,
        },
        "scores": {str(layer): scores["mean"][layer] for layer in range(32)},
        "ranking_most_to_least_sensitive": ordered,
        "most_sensitive": ordered[:3],
        "least_sensitive": ordered[-3:],
        "selected_sites": [*ordered[:3], *ordered[-3:]],
        "frozen_before_ppl": True,
    }


def proxy_agreement(selection: Mapping[str, object], validations: Sequence[Mapping[str, object]],
                    *, scores: Mapping[int, float] | None = None) -> dict[str, object]:
    impacts = {int(row["layer"]): float(row["ppl_impact_vs_phase_a_hfull"]) for row in validations}
    selected = [int(layer) for layer in selection["selected_sites"]]
    if set(impacts) != set(selected) or len(impacts) != 6:
        raise AnalysisError("proxy validation must contain the six frozen selected sites")
    if scores is None:
        scores = {int(layer): float(value) for layer, value in selection["scores"].items()}
        predicted_top = [int(layer) for layer in selection["most_sensitive"]]
        predicted_bottom = [int(layer) for layer in selection["least_sensitive"]]
    else:
        ranked_selected = sorted(selected, key=lambda layer: (-scores[layer], layer))
        predicted_top, predicted_bottom = ranked_selected[:3], ranked_selected[-3:]
    rho = spearman_rho({layer: scores[layer] for layer in selected}, impacts)
    top_mean = statistics.fmean(impacts[layer] for layer in predicted_top)
    bottom_mean = statistics.fmean(impacts[layer] for layer in predicted_bottom)
    agrees = top_mean > bottom_mean and rho > 0.0
    return {
        "predeclared_qualitative_agreement":
            "mean PPL impact of predicted top three exceeds predicted bottom three and six-site Spearman rho is positive",
        "six_site_spearman_rho": rho,
        "predicted_top_three": predicted_top, "predicted_bottom_three": predicted_bottom,
        "top_three_mean_ppl_impact": top_mean, "bottom_three_mean_ppl_impact": bottom_mean,
        "agrees": agrees,
    }


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def literal_decision(rows: Sequence[Mapping[str, object]], active_scores: Mapping[int, float],
                     *, stability_rho: float, proxy_validated: bool) -> dict[str, object]:
    indexed = _rows_by_layer_transform(rows)
    values = list(active_scores.values())
    if set(active_scores) != set(range(32)) or any(not math.isfinite(value) for value in values):
        raise AnalysisError("active sensitivity scores must cover exactly 32 finite layers")
    p10, p90 = _quantile(values, 0.10), _quantile(values, 0.90)
    scale = statistics.median(abs(value) for value in values) + 1.0e-12
    relative_spread = (p90 - p10) / scale
    quality_heterogeneous = relative_spread >= QUALITY_HETEROGENEITY_RELATIVE_SPREAD_THRESHOLD
    h32_layer = statistics.median(
        float(indexed[layer, "H32"]["timing"]["affected_packed_w4a4_layer"]["median_ms"])
        for layer in range(32)
    )
    hfull_layer = statistics.median(
        float(indexed[layer, "Hfull"]["timing"]["affected_packed_w4a4_layer"]["median_ms"])
        for layer in range(32)
    )
    gap_percent = (hfull_layer / h32_layer - 1.0) * 100.0 if h32_layer > 0 else math.inf
    latency_differs = gap_percent >= REALIZED_LATENCY_GAP_THRESHOLD_PERCENT
    ranking_stable = stability_rho >= STABILITY_THRESHOLD
    actionable_quality = quality_heterogeneous and ranking_stable and proxy_validated
    if actionable_quality and latency_differs:
        conclusion = "B1: strong heterogeneity; Phase B supports proceeding to Phase C, but Phase C was not entered"
    elif actionable_quality and not latency_differs:
        conclusion = "B2: accuracy sensitivity is heterogeneous but realized transform choices are nearly identical; kernel branch only"
    elif latency_differs:
        conclusion = "B3: realized latency differs but actionable quality sensitivity is not established; global cheaper-transform branch only"
    else:
        conclusion = "B4: neither actionable quality heterogeneity nor realized latency separation is established; stop"
    return {
        "quality_heterogeneity": {
            "statistic": "(p90 - p10) / (median(abs(sensitivity)) + 1e-12)",
            "p10": p10, "p90": p90, "relative_spread": relative_spread,
            "threshold": QUALITY_HETEROGENEITY_RELATIVE_SPREAD_THRESHOLD,
            "heterogeneous": quality_heterogeneous,
        },
        "ranking_stability": {"spearman_rho": stability_rho, "threshold": STABILITY_THRESHOLD,
                              "stable": ranking_stable},
        "realized_latency_separation": {
            "statistic": "median Hfull affected-layer latency vs median H32 affected-layer latency",
            "h32_median_ms": h32_layer, "hfull_median_ms": hfull_layer,
            "gap_percent": gap_percent, "threshold_percent": REALIZED_LATENCY_GAP_THRESHOLD_PERCENT,
            "differs": latency_differs,
        },
        "proxy_validated": proxy_validated,
        "B1": "actionable stable quality heterogeneity and >=5% realized affected-layer latency separation",
        "B2": "actionable stable quality heterogeneity without >=5% realized latency separation",
        "B3": ">=5% realized latency separation without actionable stable quality heterogeneity",
        "B4": "neither actionable quality heterogeneity nor realized latency separation",
        "conclusion": conclusion,
    }


def write_table(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("layer", "site", "transform", "normalized_output_error",
                         "activation_max_abs", "activation_rms", "transform_median_ms",
                         "affected_layer_median_ms"))
        for row in rows:
            writer.writerow((
                row["layer"], row["site"], row["transform"],
                row["quality"]["full"]["normalized_output_error"],
                row["activation_outlier"]["full"]["max_abs"],
                row["activation_outlier"]["full"]["rms"],
                row["timing"]["transform"]["median_ms"],
                row["timing"]["affected_packed_w4a4_layer"]["median_ms"],
            ))


def write_figure(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write a dependency-free SVG scatter of the 128 exact map tuples."""

    width, height = 1000, 680
    margin = {"left": 95, "right": 35, "top": 55, "bottom": 85}
    points = []
    for row in rows:
        points.append((
            float(row["timing"]["affected_packed_w4a4_layer"]["median_ms"]),
            float(row["quality"]["full"]["normalized_output_error"]),
            int(row["layer"]), str(row["transform"]),
        ))
    x_values, y_values = [p[0] for p in points], [p[1] for p in points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_span, y_span = max(x_max - x_min, 1.0e-12), max(y_max - y_min, 1.0e-12)
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    colors = {"I": "#6b7280", "H32": "#2563eb", "H128": "#7c3aed", "Hfull": "#dc2626"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#111827}.axis{stroke:#374151;stroke-width:1.5}.grid{stroke:#e5e7eb;stroke-width:1}</style>',
        '<text x="500" y="28" text-anchor="middle" font-size="20">Phase B per-layer quality–latency map</text>',
    ]
    for tick in range(6):
        fraction = tick / 5
        x = margin["left"] + fraction * plot_w
        y = margin["top"] + (1 - fraction) * plot_h
        x_value, y_value = x_min + fraction * x_span, y_min + fraction * y_span
        parts.extend((
            f'<line class="grid" x1="{x:.2f}" y1="{margin["top"]}" x2="{x:.2f}" y2="{margin["top"] + plot_h}"/>',
            f'<text x="{x:.2f}" y="{margin["top"] + plot_h + 24}" text-anchor="middle" font-size="12">{x_value:.4g}</text>',
            f'<line class="grid" x1="{margin["left"]}" y1="{y:.2f}" x2="{margin["left"] + plot_w}" y2="{y:.2f}"/>',
            f'<text x="{margin["left"] - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12">{y_value:.4g}</text>',
        ))
    parts.extend((
        f'<line class="axis" x1="{margin["left"]}" y1="{margin["top"] + plot_h}" x2="{margin["left"] + plot_w}" y2="{margin["top"] + plot_h}"/>',
        f'<line class="axis" x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{margin["top"] + plot_h}"/>',
        f'<text x="{margin["left"] + plot_w / 2}" y="{height - 24}" text-anchor="middle" font-size="15">Affected packed-W4A4 layer latency (ms)</text>',
        f'<text transform="translate(24 {margin["top"] + plot_h / 2}) rotate(-90)" text-anchor="middle" font-size="15">Local normalized output error</text>',
    ))
    for x_value, y_value, layer, transform in points:
        x = margin["left"] + (x_value - x_min) / x_span * plot_w
        y = margin["top"] + (1 - (y_value - y_min) / y_span) * plot_h
        parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{colors[transform]}" opacity="0.76">'
            f'<title>layer {layer}, {transform}: latency={x_value:.8g} ms, NMSE={y_value:.8g}</title></circle>'
        )
    for index, transform in enumerate(("I", "H32", "H128", "Hfull")):
        x = margin["left"] + 12 + index * 105
        parts.extend((
            f'<circle cx="{x}" cy="{height - 54}" r="5" fill="{colors[transform]}"/>',
            f'<text x="{x + 10}" y="{height - 49}" font-size="13">{transform}</text>',
        ))
    parts.append('</svg>')
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
