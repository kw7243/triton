#!/usr/bin/env python3
"""Fixed post-correctness Phase A timing contract."""

from __future__ import annotations

from typing import Any


SCHEMA = "vq-phase-a-timing/v1"
MANIFEST_SCHEMA = "vq-phase-a-timing-launch/v1"
RESULT_SCHEMA = "vq-phase-a-timing-result/v1"
FINAL_SCHEMA = "vq-phase-a-timing-final/v1"
CORRECTNESS_COMMIT = "8d5797fe094220f4d85ec0e74815d7a85c2405e6"
CORRECTNESS_TREE = "77a21f79ed6d8639fec234bd50ac67bb3ec97a9a"
CORRECTNESS_PARENT = "febb7c231d116da54e6b0954d7d4d82c082cfdb9"
SEED = 0
SECONDARY_SIZES = (96, 192)
ROLES = 2
KV_HEADS = 8
HEAD_DIM = 128
T_KV = 32768
DTYPE = "fp16"
VARIANTS = ("J", "F", "H")
BLOCK = 512
WARPS = 8
CONFIG_NAME = "b512-w8"
WARMUP_MS = 100
REP_MS = 500
OUTER_TRIALS = 9
QUANTILES = (0.2, 0.5, 0.8)
MAX_DISPERSION = 0.05
MAX_OUTER_STABILITY = 0.05


def snapshot() -> dict[str, Any]:
    chunks_per_head = T_KV * HEAD_DIM // 4
    chunks = ROLES * KV_HEADS * chunks_per_head
    return {
        "schema": SCHEMA,
        "mode": "post-correctness-timing-only",
        "correctness_checkpoint": {
            "commit": CORRECTNESS_COMMIT,
            "tree": CORRECTNESS_TREE,
            "parent": CORRECTNESS_PARENT,
            "classification": "PASS",
            "rerun_forbidden": True,
        },
        "seed": SEED,
        "S": list(SECONDARY_SIZES),
        "variants": list(VARIANTS),
        "dtype": DTYPE,
        "roles": ROLES,
        "kv_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "Tkv": T_KV,
        "chunks_per_head": chunks_per_head,
        "decoded_chunks_per_launch": chunks,
        "decoded_values_per_launch": chunks * 4,
        "output_bytes_per_launch": chunks * 4 * 2,
        "launch": {"block": BLOCK, "warps": WARPS, "name": CONFIG_NAME},
        "measurement": {
            "warmup_ms": WARMUP_MS,
            "rep_ms": REP_MS,
            "outer_trials": OUTER_TRIALS,
            "randomized_variant_order": True,
            "quantiles": list(QUANTILES),
            "cuda_graphs": False,
            "configuration_tuning": False,
        },
        "stability": {
            "max_within_trial_p80_p20_over_p50": MAX_DISPERSION,
            "max_outer_p80_p20_over_median_p50": MAX_OUTER_STABILITY,
            "all_J_F_H_rows_required": True,
        },
        "scope_exclusions": [
            "correctness rerun",
            "H-cache",
            "fused attention",
            "model evaluation",
            "Phase B/C/D",
            "profiling",
            "automatic retry",
        ],
    }


def benchmark_argv(python: str, stage: str, result: str) -> list[str]:
    return [
        python,
        "-B",
        f"{stage}/experiments/phase_a_decode/timing_gate/run_timing.py",
        "--output",
        result,
        "--seed",
        str(SEED),
        "--s",
        *(str(value) for value in SECONDARY_SIZES),
        "--t-kv",
        str(T_KV),
        "--roles",
        str(ROLES),
        "--kv-heads",
        str(KV_HEADS),
        "--head-dim",
        str(HEAD_DIM),
        "--dtype",
        DTYPE,
        "--config",
        CONFIG_NAME,
        "--warmup-ms",
        str(WARMUP_MS),
        "--rep-ms",
        str(REP_MS),
        "--outer-trials",
        str(OUTER_TRIALS),
    ]


def classify(rows: list[dict[str, Any]]) -> str:
    if len(rows) != len(SECONDARY_SIZES):
        return "NO RESULT"
    if [row.get("S") for row in rows] != list(SECONDARY_SIZES):
        return "NO RESULT"
    if any(not row.get("stable", False) for row in rows):
        return "NO RESULT"
    try:
        best = max(float(row["jh_speedup"]) for row in rows)
    except (KeyError, TypeError, ValueError):
        return "NO RESULT"
    if best >= 1.25:
        return "GO"
    if best >= 1.10:
        return "OPTIMIZE-ONCE"
    return "KILL"
