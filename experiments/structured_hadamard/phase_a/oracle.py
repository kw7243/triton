"""Bounded unquantized equivalence oracle at the real width 11008."""

from __future__ import annotations

import argparse
import json
import math
import random

from .reference import (D_FF, TRANSFORM_IDS, direct_forward_coordinate, fold_stored_weight, forward_rows,
                        inverse_rows, linear)
from .triton_transform import apply_transform
from .u172 import MATRIX_DIGEST, verify_orthogonality


def _random_rows(generator: random.Random, count: int) -> list[list[float]]:
    return [[generator.uniform(-0.25, 0.25) for _ in range(D_FF)] for _ in range(count)]


def _errors(actual, expected) -> tuple[float, float]:
    squared_error = 0.0
    squared_reference = 0.0
    max_abs_error = 0.0
    for actual_row, expected_row in zip(actual, expected):
        for lhs, rhs in zip(actual_row, expected_row):
            difference = float(lhs) - float(rhs)
            squared_error += difference * difference
            squared_reference += float(rhs) * float(rhs)
            max_abs_error = max(max_abs_error, abs(difference))
    return math.sqrt(squared_error / max(squared_reference, 1e-300)), max_abs_error


def _row_norm(row) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in row))


def run_oracle(*, seed: int = 0, token_rows: int = 2, weight_rows: int = 3) -> dict:
    if token_rows <= 0 or weight_rows <= 0:
        raise ValueError("token_rows and weight_rows must be positive")
    generator = random.Random(seed)
    activations = _random_rows(generator, token_rows)
    stored_weight = _random_rows(generator, weight_rows)
    bias = [generator.uniform(-0.1, 0.1) for _ in range(weight_rows)]
    baseline = linear(activations, stored_weight, bias)

    verify_orthogonality()
    results = {}
    for transform_id in TRANSFORM_IDS:
        transformed_activations = forward_rows(activations, transform_id)
        recovered = inverse_rows(transformed_activations, transform_id)
        inverse_relative, inverse_max = _errors(recovered, activations)
        norm_relative = max(abs(_row_norm(after) - _row_norm(before)) / max(_row_norm(before), 1e-300)
                            for before, after in zip(activations, transformed_activations))

        folded_weight = fold_stored_weight(stored_weight, transform_id)
        orientation_max = 0.0
        for output_index in (0, 17, D_FF - 1):
            direct = direct_forward_coordinate(stored_weight[0], transform_id, output_index)
            orientation_max = max(orientation_max, abs(folded_weight[0][output_index] - direct))

        compensated = linear(transformed_activations, folded_weight, bias)
        local_relative, local_max = _errors(compensated, baseline)
        thresholds_pass = (inverse_relative <= 1e-10 and inverse_max <= 1e-9 and norm_relative <= 1e-10
                           and orientation_max <= 1e-9 and local_relative <= 5e-5 and local_max <= 5e-4)
        if not thresholds_pass:
            raise AssertionError(
                f"{transform_id} oracle failed: inverse_rel={inverse_relative}, inverse_max={inverse_max}, "
                f"norm_rel={norm_relative}, orientation_max={orientation_max}, local_rel={local_relative}, "
                f"local_max={local_max}")
        results[transform_id] = {
            "inverse_relative_error": inverse_relative,
            "inverse_max_abs_error": inverse_max,
            "norm_relative_error": norm_relative,
            "fold_orientation_max_abs_error": orientation_max,
            "local_equivalence_relative_error": local_relative,
            "local_equivalence_max_abs_error": local_max,
            "passed": True,
        }

    if forward_rows(activations, "I") is not activations:
        raise AssertionError("CPU identity path copied its input")
    if fold_stored_weight(stored_weight, "I") is not stored_weight:
        raise AssertionError("identity weight fold copied its input")
    sentinel = object()
    if apply_transform(sentinel, "I") is not sentinel:
        raise AssertionError("Triton identity dispatch did not alias at the host boundary")

    return {
        "kind": "CPU-VALIDATION-NOT-SCIENTIFIC-EVIDENCE",
        "seed": seed,
        "shape": {"token_rows": token_rows, "weight_rows": weight_rows, "d_ff": D_FF},
        "u172_digest": MATRIX_DIGEST,
        "results": results,
        "identity_aliases": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--token-rows", type=int, default=2)
    parser.add_argument("--weight-rows", type=int, default=3)
    args = parser.parse_args(argv)
    print(json.dumps(run_oracle(seed=args.seed, token_rows=args.token_rows, weight_rows=args.weight_rows),
                     indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
