"""Create a deterministic, conspicuously synthetic Phase A schema fixture."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path
from typing import Any

from .analyze import SYNTHETIC_LABEL
from .schema import MODES, SCHEMA_VERSION, S_VALUES, TKV_VALUES, VARIANTS

FIXTURE_SEED = 20260824
CONFIGS = ("b256-w4", "b512-w8")
BASE_COMMIT = "f33a9c88651a1defe2b0a7ef4f80a3cfa1a8f25b"
WORKTREE = "/data/scratch-fast/kwen1/compute-native-vq/worktrees/vq-analysis-viz-r1"


def _dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _correctness() -> dict[str, Any]:
    metrics = {
        f"{variant}_vs_{reference}": {
            "max_abs": 0.0004 if variant == "F" else 0.0005,
            "relative_fro": 0.0002 if reference == "J" else 0.0003,
        }
        for variant in ("F", "H")
        for reference in ("J", "oracle", "quantized_oracle")
    }
    records = []
    for s_value in S_VALUES:
        for kind in ("exhaustive", "random"):
            for config in CONFIGS:
                records.append({
                    "dtype": "float16",
                    "S": s_value,
                    "input": kind,
                    "config": config,
                    "chunks": 16 * (24 * s_value if kind == "exhaustive" else 4099),
                    "metrics": metrics,
                })
    return {
        "artifact_label": SYNTHETIC_LABEL,
        "status": "passed",
        "bf16": "unsupported (synthetic fixture)",
        "records": records,
    }


def _tuning() -> dict[str, Any]:
    by_s = {}
    for s_value in S_VALUES:
        scores = {
            variant: {
                CONFIGS[0]: 0.010 + 0.001 * index + s_value / 1_000_000,
                CONFIGS[1]: 0.012 + 0.001 * index + s_value / 1_000_000,
            }
            for index, variant in enumerate(VARIANTS)
        }
        by_s[str(s_value)] = {
            "scores_ms": scores,
            "selected": {variant: CONFIGS[0] for variant in VARIANTS},
        }
    return {
        "artifact_label": SYNTHETIC_LABEL,
        "configs": list(CONFIGS),
        "selection_shape_tkv": 4096,
        "by_s": by_s,
    }


def _base_p50(s_value: int, tkv: int, variant: str, mode: str) -> float:
    scale = (tkv / 4096) ** 0.72 * (1 + s_value / 900)
    j_hot = 0.014 * scale
    hot = {"J": j_hot, "F": j_hot / 1.04, "H": j_hot / 1.12}[variant]
    return hot * (1.17 if mode == "cold" else 1.0)


def _timing_data(seed: int, unstable_cell: tuple[int, int] | None
                 ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = random.Random(seed)
    raw: dict[str, Any] = {}
    csv_rows = []
    trial_scales = (0.98, 1.01, 1.0, 1.02, 0.99)
    for s_value in S_VALUES:
        for tkv in TKV_VALUES:
            trials = []
            for trial_scale in trial_scales:
                order = list(VARIANTS)
                rng.shuffle(order)
                trial: dict[str, Any] = {"order": order}
                for variant in VARIANTS:
                    trial[variant] = {}
                    for mode in MODES:
                        spread = 0.02 + 0.005 * VARIANTS.index(variant)
                        if unstable_cell == (s_value, tkv) and variant == "H" and mode == "hot":
                            spread = 0.08
                        p50 = _base_p50(s_value, tkv, variant, mode) * trial_scale
                        trial[variant][mode] = [
                            p50 * (1 - spread / 2),
                            p50,
                            p50 * (1 + spread / 2),
                        ]
                trials.append(trial)
            aggregate = {
                variant: {
                    mode: [
                        statistics.median(trial[variant][mode][index] for trial in trials)
                        for index in range(3)
                    ]
                    for mode in MODES
                }
                for variant in VARIANTS
            }
            hot_stability = {
                variant: (
                    aggregate[variant]["hot"][2] - aggregate[variant]["hot"][0]
                ) / aggregate[variant]["hot"][1]
                for variant in ("J", "H")
            }
            attempt = {
                "rep_ms": 200,
                "trials": trials,
                "aggregate": aggregate,
                "hot_stability": hot_stability,
            }
            raw[f"S{s_value}-T{tkv}"] = [attempt]
            chunks = 2 * 8 * tkv * 128 // 4
            output_bytes = chunks * 8
            row: dict[str, Any] = {
                "S": s_value,
                "Tkv": tkv,
                "roles": 2,
                "kv_heads": 8,
                "head_dim": 128,
                "chunks": chunks,
                "output_bytes": output_bytes,
                "warmup_ms": 25,
                "rep_ms": 200,
                "outer_trials": 5,
                "j_config": CONFIGS[0],
                "f_config": CONFIGS[0],
                "h_config": CONFIGS[0],
            }
            for variant in VARIANTS:
                for mode in MODES:
                    p20, p50, p80 = aggregate[variant][mode]
                    prefix = f"{variant.lower()}_{mode}"
                    seconds = p50 / 1000
                    row.update({
                        f"{prefix}_p20_ms": p20,
                        f"{prefix}_p50_ms": p50,
                        f"{prefix}_p80_ms": p80,
                        f"{prefix}_gchunks_s": chunks / seconds / 1e9,
                        f"{prefix}_output_gib_s": output_bytes / seconds / 2**30,
                        f"{prefix}_stability": (p80 - p20) / p50,
                    })
            row["jh_hot_speedup"] = row["j_hot_p50_ms"] / row["h_hot_p50_ms"]
            row["jh_cold_speedup"] = row["j_cold_p50_ms"] / row["h_cold_p50_ms"]
            row["stable"] = max(row["j_hot_stability"], row["h_hot_stability"]) <= 0.05
            row.update({
                "f_max_abs_vs_j": 0.0004,
                "f_relative_fro_vs_j": 0.0002,
                "f_max_abs_vs_oracle": 0.0006,
                "f_relative_fro_vs_oracle": 0.0003,
                "h_max_abs_vs_j": 0.0005,
                "h_relative_fro_vs_j": 0.0002,
                "h_max_abs_vs_oracle": 0.0007,
                "h_relative_fro_vs_oracle": 0.0003,
                "h_max_abs": 0.0007,
                "h_relative_fro": 0.0003,
                "decision": SYNTHETIC_LABEL,
            })
            csv_rows.append(row)
    return raw, csv_rows


def build_fixture(output: Path, seed: int = FIXTURE_SEED,
                  unstable_cell: tuple[int, int] | None = None) -> None:
    """Write a complete fixture, refusing to replace any existing evidence."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty fixture directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    raw, rows = _timing_data(seed, unstable_cell)
    _dump(output / "correctness.json", _correctness())
    _dump(output / "tuning.json", _tuning())
    _dump(output / "trial_timings.json", raw)
    _dump(output / "run_metadata.json", {
        "artifact_label": SYNTHETIC_LABEL,
        "status": "passed",
        "git_commit": BASE_COMMIT,
        "git_branch": "fm/phase-a-analysis-viz",
        "cwd": WORKTREE,
        "torch": "2.13.0+synthetic",
        "triton": "3.7.1+synthetic",
        "cuda_runtime": "13.0-synthetic",
        "device": "SYNTHETIC GPU — NO DEVICE WAS USED",
        "compute_capability": [8, 6],
        "assumptions": {
            "quaternion": "scalar-first (w,x,y,z)",
            "id": "p*S+s",
        },
        "synthetic": True,
        "fixture_seed": seed,
        "schema_version": SCHEMA_VERSION,
    })
    with (output / "timings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=FIXTURE_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_fixture(args.output, args.seed)
    print(args.output)


if __name__ == "__main__":
    main()
