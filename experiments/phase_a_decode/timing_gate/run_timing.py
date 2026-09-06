#!/usr/bin/env python3
"""Run the two-row, timing-only J/F/H Phase A gate."""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PHASE_A = HERE.parent
REPO = PHASE_A.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PHASE_A))
sys.path.insert(0, str(REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import triton

from benchmark import launch, make_tables
from contract import (
    BLOCK,
    CONFIG_NAME,
    DTYPE,
    HEAD_DIM,
    KV_HEADS,
    OUTER_TRIALS,
    QUANTILES,
    REP_MS,
    RESULT_SCHEMA,
    ROLES,
    SECONDARY_SIZES,
    SEED,
    T_KV,
    VARIANTS,
    WARMUP_MS,
    WARPS,
    classify,
    snapshot,
)
from runtime import atomic_json, atomic_text, utc_now


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def measure_size(size: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    chunks_per_head = T_KV * HEAD_DIM // 4
    chunks = ROLES * KV_HEADS * chunks_per_head
    tables = make_tables(SEED, ROLES, KV_HEADS, size, torch.float16)
    generator = torch.Generator().manual_seed(SEED + size)
    ids = torch.randint(24 * size, (chunks,), dtype=torch.int32, generator=generator).cuda()
    output = torch.empty((chunks, 4), dtype=torch.float16, device="cuda")
    config = (BLOCK, WARPS, CONFIG_NAME)
    functions = {
        variant: (
            lambda variant=variant: launch(
                variant, ids, tables, output, chunks_per_head, size, config
            )
        )
        for variant in VARIANTS
    }
    for function in functions.values():
        function()
    torch.cuda.synchronize()

    rng = random.Random(SEED + size * 100003 + T_KV)
    trials: list[dict[str, Any]] = []
    for trial_index in range(OUTER_TRIALS):
        order = list(VARIANTS)
        rng.shuffle(order)
        trial: dict[str, Any] = {"trial": trial_index, "order": order, "variants": {}}
        for variant in order:
            measured = triton.testing.do_bench(
                functions[variant],
                warmup=WARMUP_MS,
                rep=REP_MS,
                quantiles=list(QUANTILES),
            )
            p20, p50, p80 = (float(value) * 1000.0 for value in measured)
            trial["variants"][variant] = {
                "p20_us": p20,
                "p50_us": p50,
                "p80_us": p80,
            }
        trials.append(trial)

    variants: dict[str, Any] = {}
    for variant in VARIANTS:
        p20s = [trial["variants"][variant]["p20_us"] for trial in trials]
        p50s = [trial["variants"][variant]["p50_us"] for trial in trials]
        p80s = [trial["variants"][variant]["p80_us"] for trial in trials]
        p20 = statistics.median(p20s)
        p50 = statistics.median(p50s)
        p80 = statistics.median(p80s)
        outer_p20 = quantile(p50s, 0.2)
        outer_p80 = quantile(p50s, 0.8)
        dispersion = (p80 - p20) / p50
        outer_stability = (outer_p80 - outer_p20) / p50
        variants[variant] = {
            "p20_us": p20,
            "p50_us": p50,
            "p80_us": p80,
            "dispersion": dispersion,
            "outer_p50_min_us": min(p50s),
            "outer_p50_max_us": max(p50s),
            "outer_stability": outer_stability,
            "outer_rsd": (
                statistics.pstdev(p50s) / statistics.mean(p50s)
                if statistics.mean(p50s)
                else 0.0
            ),
            "gchunks_s": chunks / (p50 / 1e6) / 1e9,
            "output_gib_s": (chunks * 4 * 2) / (p50 / 1e6) / 2**30,
            "stable": dispersion <= 0.05 and outer_stability <= 0.05,
        }
    row = {
        "S": size,
        "Tkv": T_KV,
        "roles": ROLES,
        "kv_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "dtype": DTYPE,
        "chunks": chunks,
        "config": CONFIG_NAME,
        "variants": variants,
        "jh_speedup": variants["J"]["p50_us"] / variants["H"]["p50_us"],
        "stable": all(variants[variant]["stable"] for variant in VARIANTS),
    }
    del ids, output, tables
    torch.cuda.empty_cache()
    return row, trials


def write_outputs(
    output: Path,
    rows: list[dict[str, Any]],
    trials: dict[str, Any],
    manifest_digest: str,
) -> str:
    decision = classify(rows)
    payload = {
        "schema": RESULT_SCHEMA,
        "recorded_at_utc": utc_now(),
        "launch_manifest_sha256": manifest_digest,
        "classification": decision,
        "contract": snapshot(),
        "rows": rows,
    }
    atomic_json(output / "timing_output.json", payload)
    atomic_json(
        output / "trial_timings.json",
        {
            "schema": "vq-phase-a-timing-trials/v1",
            "launch_manifest_sha256": manifest_digest,
            "trials": trials,
        },
    )
    fields = [
        "S",
        "Tkv",
        "J_p50_us",
        "J_dispersion",
        "J_stability",
        "F_p50_us",
        "F_dispersion",
        "F_stability",
        "H_p50_us",
        "H_dispersion",
        "H_stability",
        "JH_speedup",
        "stable",
        "classification",
        "manifest_sha256",
    ]
    lines = [",".join(fields)]
    for row in rows:
        values: list[Any] = [row["S"], row["Tkv"]]
        for variant in VARIANTS:
            values.extend(
                [
                    f"{row['variants'][variant]['p50_us']:.6f}",
                    f"{row['variants'][variant]['dispersion']:.8f}",
                    f"{row['variants'][variant]['outer_stability']:.8f}",
                ]
            )
        values.extend(
            [
                f"{row['jh_speedup']:.8f}",
                str(row["stable"]).lower(),
                decision,
                manifest_digest,
            ]
        )
        lines.append(",".join(map(str, values)))
    atomic_text(output / "timings.csv", "\n".join(lines) + "\n")

    table = [
        "# Phase A timing gate",
        "",
        f"Decision: **{decision}**",
        "",
        "| S | J latency (us) | J disp./stability | F latency (us) | F disp./stability | H latency (us) | H disp./stability | J/H |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        cells = [str(row["S"])]
        for variant in VARIANTS:
            metric = row["variants"][variant]
            cells.extend(
                [
                    f"{metric['p50_us']:.3f}",
                    f"{100 * metric['dispersion']:.2f}% / {100 * metric['outer_stability']:.2f}%",
                ]
            )
        cells.append(f"{row['jh_speedup']:.3f}x")
        table.append("| " + " | ".join(cells) + " |")
    table.extend(
        [
            "",
            "Latency is the median of nine randomized outer-trial p50 values. "
            "Dispersion is median within-trial (p80-p20)/p50; stability is the "
            "outer-trial p80-p20 span divided by the median p50.",
            "",
        ]
    )
    atomic_text(output / "table.md", "\n".join(table))

    fig, axis = plt.subplots(figsize=(7.2, 4.3))
    positions = list(range(len(rows)))
    width = 0.24
    colors = {"J": "#355C7D", "F": "#C06C84", "H": "#2A9D8F"}
    for offset, variant in enumerate(VARIANTS):
        values = [row["variants"][variant]["p50_us"] for row in rows]
        low = [
            row["variants"][variant]["p50_us"] - row["variants"][variant]["p20_us"]
            for row in rows
        ]
        high = [
            row["variants"][variant]["p80_us"] - row["variants"][variant]["p50_us"]
            for row in rows
        ]
        axis.bar(
            [position + (offset - 1) * width for position in positions],
            values,
            width,
            yerr=[low, high],
            capsize=3,
            label=variant,
            color=colors[variant],
        )
    axis.set_xticks(positions, [f"S={row['S']}" for row in rows])
    axis.set_ylabel("Kernel latency (us)")
    axis.set_title("Compute-Native VQ Phase A, fp16 Tkv=32768")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    temporary = output / f".latency-{os.getpid()}.png"
    fig.savefig(temporary, dpi=180)
    plt.close(fig)
    os.replace(temporary, output / "latency.png")
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--s", type=int, nargs="+", required=True)
    parser.add_argument("--t-kv", type=int, required=True)
    parser.add_argument("--roles", type=int, required=True)
    parser.add_argument("--kv-heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--warmup-ms", type=int, required=True)
    parser.add_argument("--rep-ms", type=int, required=True)
    parser.add_argument("--outer-trials", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    observed = (
        args.seed,
        tuple(args.s),
        args.t_kv,
        args.roles,
        args.kv_heads,
        args.head_dim,
        args.dtype,
        args.config,
        args.warmup_ms,
        args.rep_ms,
        args.outer_trials,
    )
    expected = (
        SEED,
        SECONDARY_SIZES,
        T_KV,
        ROLES,
        KV_HEADS,
        HEAD_DIM,
        DTYPE,
        CONFIG_NAME,
        WARMUP_MS,
        REP_MS,
        OUTER_TRIALS,
    )
    if observed != expected:
        raise ValueError(f"timing contract drift: {observed!r} != {expected!r}")
    digest = os.environ.get("PHASE_A_MANIFEST_SHA256", "")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise RuntimeError("immutable launch manifest digest is missing")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA device is required")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    rows: list[dict[str, Any]] = []
    trials: dict[str, Any] = {}
    for size in SECONDARY_SIZES:
        print(f"timing S={size} Tkv={T_KV}", flush=True)
        row, current = measure_size(size)
        rows.append(row)
        trials[f"S{size}"] = current
    decision = write_outputs(args.output, rows, trials, digest)
    atomic_json(
        args.output / "run_metadata.json",
        {
            "schema": "vq-phase-a-timing-run/v1",
            "status": "complete",
            "classification": decision,
            "launch_manifest_sha256": digest,
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "torch": torch.__version__,
            "triton": triton.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "correctness_rerun": False,
            "configuration_tuning": False,
            "cuda_graphs": False,
        },
    )
    print(f"classification={decision}", flush=True)
    return 0 if decision != "NO RESULT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
