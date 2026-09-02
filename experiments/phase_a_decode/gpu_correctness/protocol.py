#!/usr/bin/env python3
"""Standard-library definition of the Phase A correctness-only contract."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Sequence


SEED = 0
SECONDARY_SIZES = (96, 192)
ROLES = 2
KV_HEADS = 8
HEAD_DIM = 128
PRIMARY_UNITS = 24
RANDOM_IDS_PER_ROLE_HEAD = 4099
INPUT_KINDS = ("exhaustive", "random")
CONFIGS = ((256, 4, "b256-w4"), (512, 8, "b512-w8"))
ABS_TOL = 4e-3
REL_TOL = 1e-3
FORBIDDEN_ARTIFACTS = (
    "timings.csv",
    "tuning.json",
    "trial_timings.json",
    "jh_speedup.png",
)


class ScientificComparisonFailure(AssertionError):
    """A finite, bitwise, or declared numerical comparison failed."""

    def __init__(self, check: str, detail: str):
        super().__init__(f"{check}: {detail}")
        self.check = check


@dataclass(frozen=True)
class CorrectnessArguments:
    correctness_only: bool
    seed: int
    secondary_sizes: tuple[int, ...]
    roles: int
    kv_heads: int
    head_dim: int


def validate_correctness_arguments(arguments: CorrectnessArguments) -> None:
    """Reject any workload drift or attempt to enter a timing-capable mode."""

    errors = []
    if not arguments.correctness_only:
        errors.append("--correctness-only is required")
    if arguments.seed != SEED:
        errors.append(f"seed must be {SEED}")
    if tuple(arguments.secondary_sizes) != SECONDARY_SIZES:
        errors.append(f"S values must be ordered exactly as {SECONDARY_SIZES}")
    if arguments.roles != ROLES:
        errors.append(f"roles must be {ROLES}")
    if arguments.kv_heads != KV_HEADS:
        errors.append(f"kv_heads must be {KV_HEADS}")
    if arguments.head_dim != HEAD_DIM:
        errors.append(f"head_dim must be {HEAD_DIM}")
    if errors:
        raise ValueError("invalid Phase A correctness-only protocol: " + "; ".join(errors))


def chunks_per_record(secondary_size: int, input_kind: str) -> int:
    if secondary_size not in SECONDARY_SIZES:
        raise ValueError(f"unsupported secondary size: {secondary_size}")
    if input_kind == "exhaustive":
        per_role_head = PRIMARY_UNITS * secondary_size
    elif input_kind == "random":
        per_role_head = RANDOM_IDS_PER_ROLE_HEAD
    else:
        raise ValueError(f"unsupported input kind: {input_kind}")
    return ROLES * KV_HEADS * per_role_head


def contract_snapshot(*, bf16_supported: bool | None = None) -> dict[str, object]:
    records_per_dtype = len(SECONDARY_SIZES) * len(INPUT_KINDS) * len(CONFIGS)
    expected_records = (
        records_per_dtype * (2 if bf16_supported else 1)
        if bf16_supported is not None
        else None
    )
    return {
        "mode": "correctness-only",
        "seed": SEED,
        "S": list(SECONDARY_SIZES),
        "roles": ROLES,
        "kv_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "primary_units": PRIMARY_UNITS,
        "inputs": list(INPUT_KINDS),
        "configs": [config[2] for config in CONFIGS],
        "tolerances": {"absolute": ABS_TOL, "relative_frobenius": REL_TOL},
        "records_per_dtype": records_per_dtype,
        "expected_fp16_records": records_per_dtype,
        "expected_bf16_records_if_supported": records_per_dtype,
        "expected_records": expected_records,
        "bf16_supported": bf16_supported,
        "chunks": {
            f"S{secondary_size}-{kind}": chunks_per_record(secondary_size, kind)
            for secondary_size in SECONDARY_SIZES
            for kind in INPUT_KINDS
        },
        "bf16_direct_float32_threshold": "recorded-not-enforced",
        "timing_allowed": False,
        "forbidden_artifacts": list(FORBIDDEN_ARTIFACTS),
    }


def benchmark_argv(python: str, staged_repo: str, result_dir: str) -> list[str]:
    return [
        python,
        f"{staged_repo}/experiments/phase_a_decode/benchmark.py",
        "--correctness-only",
        "--seed",
        str(SEED),
        "--s",
        *(str(value) for value in SECONDARY_SIZES),
        "--roles",
        str(ROLES),
        "--kv-heads",
        str(KV_HEADS),
        "--head-dim",
        str(HEAD_DIM),
        "--output",
        result_dir,
    ]


def _parse_cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-contract", action="store_true", required=True)
    parser.add_argument("--bf16-supported", choices=("yes", "no", "unknown"), default="unknown")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_cli(argv)
    bf16 = {"yes": True, "no": False, "unknown": None}[args.bf16_supported]
    print(json.dumps(contract_snapshot(bf16_supported=bf16), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
