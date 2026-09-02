"""Dependency-free reference semantics for Phase B block Hadamards."""

from __future__ import annotations

import math
from typing import Sequence


TRANSFORMS = ("I", "H32", "H128", "Hfull")
BLOCK_SIZES = {"I": None, "H32": 32, "H128": 128, "Hfull": None}


class ReferenceError(ValueError):
    """A transform request does not have the frozen Phase B semantics."""


def _fht_in_place(values: list[float]) -> None:
    if not values or len(values) & (len(values) - 1):
        raise ReferenceError("Hadamard block width must be a positive power of two")
    half = 1
    while half < len(values):
        for base in range(0, len(values), 2 * half):
            for offset in range(half):
                lhs = values[base + offset]
                rhs = values[base + half + offset]
                values[base + offset] = lhs + rhs
                values[base + half + offset] = lhs - rhs
        half *= 2


def block_hadamard_rows(rows: Sequence[Sequence[float]], block_size: int) -> list[list[float]]:
    """Apply normalized Sylvester Hadamards to consecutive feature blocks."""

    if type(block_size) is not int or block_size <= 0 or block_size & (block_size - 1):
        raise ReferenceError("block_size must be a positive power of two")
    if not rows or not rows[0]:
        raise ReferenceError("rows must be a nonempty rectangular matrix")
    width = len(rows[0])
    if width % block_size or any(len(row) != width for row in rows):
        raise ReferenceError("row width must be rectangular and divisible by block_size")
    scale = 1.0 / math.sqrt(block_size)
    result = []
    for row in rows:
        output = []
        for begin in range(0, width, block_size):
            block = [float(value) for value in row[begin:begin + block_size]]
            _fht_in_place(block)
            output.extend(value * scale for value in block)
        result.append(output)
    return result


def fold_block_weight_rows(rows: Sequence[Sequence[float]], block_size: int) -> list[list[float]]:
    """Fold the same symmetric, orthonormal block transform into weight rows."""

    return block_hadamard_rows(rows, block_size)


def block_transform_invariants(block_size: int) -> dict[str, float]:
    """Bounded deterministic inverse and folded-linear equivalence reference."""

    width = 2 * block_size
    activation = [[((index * 7) % 19 - 9) / 8.0 for index in range(width)]]
    weights = [
        [((row * 11 + index * 5) % 23 - 11) / 7.0 for index in range(width)]
        for row in range(3)
    ]
    transformed = block_hadamard_rows(activation, block_size)
    recovered = block_hadamard_rows(transformed, block_size)
    folded = fold_block_weight_rows(weights, block_size)
    baseline = [sum(x * w for x, w in zip(activation[0], row)) for row in weights]
    compensated = [sum(x * w for x, w in zip(transformed[0], row)) for row in folded]
    return {
        "inverse_max_abs": max(abs(a - b) for a, b in zip(activation[0], recovered[0])),
        "equivalence_max_abs": max(abs(a - b) for a, b in zip(baseline, compensated)),
    }


def block_semantics(transform: str, width: int, *, full_outer_order: int | None = None) -> dict[str, object]:
    if transform not in TRANSFORMS:
        raise ReferenceError(f"unknown transform: {transform}")
    if type(width) is not int or width <= 0:
        raise ReferenceError("width must be a positive integer")
    if transform == "I":
        return {
            "family": "identity", "block_size": None, "blocks": 0,
            "normalization": "none", "layout": "host alias; no arithmetic",
            "permutation": "none", "random_signs": "none",
        }
    if transform in ("H32", "H128"):
        block_size = BLOCK_SIZES[transform]
        assert block_size is not None
        if width % block_size:
            raise ReferenceError(f"width {width} is not divisible by {block_size}")
        return {
            "family": "block_hadamard", "block_size": block_size,
            "blocks": width // block_size, "normalization": f"1/sqrt({block_size})",
            "layout": "consecutive non-overlapping feature blocks",
            "construction": "normalized Sylvester FWHT independently per block",
            "weight_folding": "same symmetric orthonormal block transform on each weight row",
            "permutation": "none", "random_signs": "none",
        }
    if not full_outer_order or width % full_outer_order:
        raise ReferenceError("Hfull requires a valid pinned QuaRot outer order")
    inner = width // full_outer_order
    if inner & (inner - 1):
        raise ReferenceError("Hfull inner order must be a power of two")
    return {
        "family": "full_hadamard", "block_size": width, "blocks": 1,
        "normalization": f"1/sqrt({width})",
        "layout": "pinned QuaRot H_outer tensor H_inner factorization",
        "outer_order": full_outer_order, "inner_order": inner,
        "weight_folding": "same pinned full transform on each weight row",
        "permutation": "none", "random_signs": "none",
    }
