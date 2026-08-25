"""Validate Phase A results and emit descriptive tables plus one SVG plot.

This module never emits a GO/OPTIMIZE/KILL recommendation.  It reports only
validated measurements, ratios, and observed timing spread.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any

from .schema import MODES, REQUIRED_FILES, SCHEMA_VERSION, VARIANTS, validate_result_directory

SYNTHETIC_LABEL = "SYNTHETIC — NOT A SCIENTIFIC RESULT"
REAL_LABEL = "DESCRIPTIVE ANALYSIS — NO GATE DECISION"
OUTPUT_FILES = (
    "analysis_table.csv",
    "analysis_table.md",
    "comparison.svg",
    "validation_report.json",
)


def aggregate(validated: dict[str, Any]) -> list[dict[str, Any]]:
    """Compute deterministic J/H, F/H, and stability summaries."""
    rows = []
    for cell in sorted(validated["timings"]):
        source = validated["timings"][cell]
        row: dict[str, Any] = {"S": cell[0], "Tkv": cell[1]}
        for variant in VARIANTS:
            for mode in MODES:
                prefix = f"{variant.lower()}_{mode}"
                row[f"{prefix}_p50_us"] = source[f"{prefix}_quantiles_ms"][1] * 1000
                row[f"{prefix}_stability"] = source[f"{prefix}_stability"]
        for mode in MODES:
            h_p50 = source[f"h_{mode}_quantiles_ms"][1]
            row[f"j_over_h_{mode}"] = source[f"j_{mode}_quantiles_ms"][1] / h_p50
            row[f"f_over_h_{mode}"] = source[f"f_{mode}_quantiles_ms"][1] / h_p50
        row["max_hot_stability"] = max(
            source[f"{variant.lower()}_hot_stability"] for variant in VARIANTS
        )
        row["stability_summary"] = (
            "within 5% spread"
            if row["max_hot_stability"] <= 0.05
            else "UNSTABLE — DESCRIPTIVE ONLY"
        )
        rows.append(row)
    return rows


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _label(validated: dict[str, Any]) -> str:
    return SYNTHETIC_LABEL if validated["provenance"]["synthetic"] else REAL_LABEL


def _csv_rows(rows: list[dict[str, Any]], label: str) -> tuple[list[str], list[dict[str, str]]]:
    fields = [
        "artifact_label", "S", "Tkv", "j_hot_p50_us", "f_hot_p50_us", "h_hot_p50_us",
        "j_over_h_hot", "f_over_h_hot", "j_over_h_cold", "f_over_h_cold",
        "j_hot_stability", "f_hot_stability", "h_hot_stability",
        "max_hot_stability", "stability_summary",
    ]
    output = []
    for row in rows:
        output.append({
            "artifact_label": label,
            "S": str(row["S"]),
            "Tkv": str(row["Tkv"]),
            "j_hot_p50_us": _fmt(row["j_hot_p50_us"], 3),
            "f_hot_p50_us": _fmt(row["f_hot_p50_us"], 3),
            "h_hot_p50_us": _fmt(row["h_hot_p50_us"], 3),
            "j_over_h_hot": _fmt(row["j_over_h_hot"]),
            "f_over_h_hot": _fmt(row["f_over_h_hot"]),
            "j_over_h_cold": _fmt(row["j_over_h_cold"]),
            "f_over_h_cold": _fmt(row["f_over_h_cold"]),
            "j_hot_stability": _fmt(row["j_hot_stability"]),
            "f_hot_stability": _fmt(row["f_hot_stability"]),
            "h_hot_stability": _fmt(row["h_hot_stability"]),
            "max_hot_stability": _fmt(row["max_hot_stability"]),
            "stability_summary": row["stability_summary"],
        })
    return fields, output


def render_csv(rows: list[dict[str, Any]], label: str) -> str:
    import io

    fields, normalized = _csv_rows(rows, label)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(normalized)
    return stream.getvalue()


def render_markdown(rows: list[dict[str, Any]], label: str) -> str:
    lines = [
        f"# {label}",
        "",
        "Descriptive Phase A comparisons only. Ratios are baseline latency divided by H latency; "
        "values above 1 mean H had lower latency in the supplied measurements.",
        "",
        "| S | Tkv | J hot µs | F hot µs | H hot µs | J/H hot | F/H hot | "
        "J/H cold | F/H cold | max hot spread | stability |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['S']} | {row['Tkv']} | {_fmt(row['j_hot_p50_us'], 3)} | "
            f"{_fmt(row['f_hot_p50_us'], 3)} | {_fmt(row['h_hot_p50_us'], 3)} | "
            f"{_fmt(row['j_over_h_hot'])} | {_fmt(row['f_over_h_hot'])} | "
            f"{_fmt(row['j_over_h_cold'])} | {_fmt(row['f_over_h_cold'])} | "
            f"{_fmt(row['max_hot_stability'])} | {row['stability_summary']} |"
        )
    lines += [
        "",
        f"Watermark: **{label}**",
        "",
        "This lane does not infer a launch gate or a weight-VQ pivot.",
    ]
    return "\n".join(lines) + "\n"


def render_svg(rows: list[dict[str, Any]], label: str) -> str:
    width, height = 900, 520
    left, right, top, bottom = 82, 35, 88, 76
    plot_width, plot_height = width - left - right, height - top - bottom
    values = [
        row[field]
        for row in rows
        for field in ("j_over_h_hot", "f_over_h_hot")
    ]
    low = min(0.95, min(values) - 0.04)
    high = max(1.05, max(values) + 0.04)
    if math.isclose(low, high):
        high = low + 0.1

    def x_pos(tkv: int) -> float:
        lo, hi = math.log2(4096), math.log2(32768)
        return left + (math.log2(tkv) - lo) / (hi - lo) * plot_width

    def y_pos(value: float) -> float:
        return top + (high - value) / (high - low) * plot_height

    colors = {96: "#2563eb", 192: "#dc2626"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Phase A J/H and F/H hot timing ratios</title>',
        f'<desc id="desc">{html.escape(label)}. Descriptive timing ratios only.</desc>',
        '<rect width="100%" height="100%" fill="#fffdf8"/>',
        '<style>text{font-family:ui-sans-serif,system-ui,sans-serif;fill:#172033}'
        '.axis{stroke:#536174;stroke-width:1}.grid{stroke:#d9dee7;stroke-width:1}'
        '.series{fill:none;stroke-width:3}.point{stroke:#fffdf8;stroke-width:2}'
        '.small{font-size:13px}.tick{font-size:12px}.legend{font-size:13px}</style>',
        f'<text x="{left}" y="32" font-size="22" font-weight="700">Phase A descriptive comparisons</text>',
        f'<text x="{left}" y="56" font-size="14">{html.escape(label)}</text>',
        f'<text x="{left}" y="75" font-size="12">Hot p50 ratio; baseline / H. No gate decision.</text>',
    ]
    for index in range(6):
        value = low + index * (high - low) / 5
        y = y_pos(value)
        parts += [
            f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>',
            f'<text class="tick" x="{left-10}" y="{y+4:.2f}" text-anchor="end">{value:.2f}</text>',
        ]
    baseline_y = y_pos(1.0)
    parts.append(
        f'<line x1="{left}" y1="{baseline_y:.2f}" x2="{width-right}" y2="{baseline_y:.2f}" '
        'stroke="#111827" stroke-width="1.5"/>'
    )
    for tkv, text in ((4096, "4K"), (16384, "16K"), (32768, "32K")):
        x = x_pos(tkv)
        parts += [
            f'<line class="axis" x1="{x:.2f}" y1="{height-bottom}" x2="{x:.2f}" y2="{height-bottom+6}"/>',
            f'<text class="tick" x="{x:.2f}" y="{height-bottom+25}" text-anchor="middle">{text}</text>',
        ]
    parts += [
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>',
        f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"/>',
        f'<text class="small" x="{left + plot_width/2:.2f}" y="{height-18}" text-anchor="middle">Tkv</text>',
        f'<text class="small" transform="translate(18 {top+plot_height/2:.2f}) rotate(-90)" '
        'text-anchor="middle">latency ratio</text>',
    ]
    for s_value in (96, 192):
        chosen = [row for row in rows if row["S"] == s_value]
        for field, dash, marker in (
            ("j_over_h_hot", "", "circle"),
            ("f_over_h_hot", ' stroke-dasharray="8 5"', "square"),
        ):
            points = " ".join(
                f"{x_pos(row['Tkv']):.2f},{y_pos(row[field]):.2f}" for row in chosen
            )
            parts.append(
                f'<polyline class="series" stroke="{colors[s_value]}"{dash} points="{points}"/>'
            )
            for row in chosen:
                x, y = x_pos(row["Tkv"]), y_pos(row[field])
                if marker == "circle":
                    parts.append(
                        f'<circle class="point" cx="{x:.2f}" cy="{y:.2f}" r="5" '
                        f'fill="{colors[s_value]}"/>'
                    )
                else:
                    parts.append(
                        f'<rect class="point" x="{x-4.5:.2f}" y="{y-4.5:.2f}" width="9" height="9" '
                        f'fill="{colors[s_value]}"/>'
                    )
    legend_x = width - 310
    for index, (s_value, field_label, dash) in enumerate((
        (96, "S=96 J/H", ""),
        (96, "S=96 F/H", ' stroke-dasharray="8 5"'),
        (192, "S=192 J/H", ""),
        (192, "S=192 F/H", ' stroke-dasharray="8 5"'),
    )):
        y = 22 + index * 18
        parts += [
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+30}" y2="{y}" '
            f'stroke="{colors[s_value]}" stroke-width="3"{dash}/>',
            f'<text class="legend" x="{legend_x+38}" y="{y+4}">{field_label}</text>',
        ]
    parts += [
        f'<text x="{left+plot_width/2:.2f}" y="{top+plot_height/2:.2f}" '
        'text-anchor="middle" font-size="28" font-weight="700" fill="#64748b" '
        'opacity="0.13" transform="rotate(-18 '
        f'{left+plot_width/2:.2f} {top+plot_height/2:.2f})">{html.escape(label)}</text>',
        "</svg>",
    ]
    return "\n".join(parts) + "\n"


def _hash_inputs(input_dir: Path) -> dict[str, str]:
    hashes = {}
    for name in REQUIRED_FILES:
        hashes[name] = hashlib.sha256((input_dir / name).read_bytes()).hexdigest()
    return hashes


def write_artifacts(input_dir: Path, output_dir: Path, overwrite: bool = False) -> list[Path]:
    validated = validate_result_directory(input_dir)
    rows = aggregate(validated)
    label = _label(validated)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / name for name in OUTPUT_FILES]
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite generated artifacts: " + ", ".join(str(path) for path in existing)
        )
    payloads = {
        "analysis_table.csv": render_csv(rows, label),
        "analysis_table.md": render_markdown(rows, label),
        "comparison.svg": render_svg(rows, label),
        "validation_report.json": json.dumps(
            {
                "artifact_label": label,
                "schema_version": SCHEMA_VERSION,
                "validation": "passed",
                "scientific_decision": None,
                "synthetic": validated["provenance"]["synthetic"],
                "fixture_seed": validated["provenance"]["fixture_seed"],
                "input_sha256": _hash_inputs(input_dir),
                "correctness_cases": validated["correctness"]["cases"],
                "timing_cells": len(rows),
                "unstable_cells": [
                    {"S": row["S"], "Tkv": row["Tkv"]}
                    for row in rows
                    if row["max_hot_stability"] > 0.05
                ],
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
    }
    for name, payload in payloads.items():
        (output_dir / name).write_text(payload, encoding="utf-8")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="directory containing Phase A inputs")
    parser.add_argument("--output", type=Path, required=True, help="directory for descriptive artifacts")
    parser.add_argument("--overwrite", action="store_true", help="replace only this tool's known outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = write_artifacts(args.input, args.output, args.overwrite)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
