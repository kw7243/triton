"""Dependency-free CPU contract for the Phase A transform menu.

Rows use the right-transform convention. For a stored PyTorch linear weight
``A`` with shape ``[out_features, in_features]``, the compensated fold is the
same forward row operator: ``A_t = A R_t``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .u172 import MATRIX_DIGEST, u172

D_FF = 11008
D_MODEL = 4096
FULL_K = 172
FULL_Q = 64
TRANSFORM_IDS = ("I", "H32", "H128", "Hfull")


@dataclass(frozen=True)
class TransformSpec:
    id: str
    d: int
    block_size: int | None
    K: int | None
    q: int | None
    normalization: str
    axis: str
    channel_order: str
    sign: str
    permutation: str
    matrix_digest: str | None
    implementation: str


_SPECS = {
    "I": TransformSpec("I", D_FF, None, None, None, "identity", "last/right", "contiguous-natural",
                       "none", "identity", None, "host-no-launch"),
    "H32": TransformSpec("H32", D_FF, 32, None, None, "orthonormal", "last/right", "contiguous-natural",
                         "none", "identity", None, "cpu-reference-sylvester"),
    "H128": TransformSpec("H128", D_FF, 128, None, None, "orthonormal", "last/right", "contiguous-natural",
                          "none", "identity", None, "cpu-reference-sylvester"),
    "Hfull": TransformSpec("Hfull", D_FF, None, FULL_K, FULL_Q, "orthonormal", "last/right",
                           "contiguous-natural", "none", "identity", MATRIX_DIGEST, "triton-fht64-u172"),
}


def transform_spec(transform_id: str) -> TransformSpec:
    try:
        return _SPECS[transform_id]
    except KeyError as exc:
        raise ValueError(f"unknown transform {transform_id!r}; expected one of {TRANSFORM_IDS}") from exc


def _validate_rows(rows: Sequence[Sequence[float]]) -> int:
    if not hasattr(rows, "__len__"):
        raise TypeError("rows must be a sized sequence")
    if len(rows) == 0:
        raise ValueError("at least one row is required")
    width = len(rows[0])
    if width != D_FF:
        raise ValueError(f"last dimension must be exactly {D_FF}, got {width}")
    if any(len(row) != width for row in rows):
        raise ValueError("rows must be rectangular")
    return width


def _fht_in_place(values: list[float]) -> None:
    """Unnormalized natural-order Sylvester transform."""

    size = len(values)
    if size == 0 or size & (size - 1):
        raise ValueError(f"FHT size must be a positive power of two, got {size}")
    half = 1
    while half < size:
        group = half * 2
        for base in range(0, size, group):
            for offset in range(half):
                lhs = values[base + offset]
                rhs = values[base + half + offset]
                values[base + offset] = lhs + rhs
                values[base + half + offset] = lhs - rhs
        half = group


def _block_forward(rows: Sequence[Sequence[float]], block_size: int) -> list[list[float]]:
    scale = 1.0 / math.sqrt(block_size)
    transformed: list[list[float]] = []
    for source in rows:
        target = [float(value) for value in source]
        for base in range(0, D_FF, block_size):
            block = target[base:base + block_size]
            _fht_in_place(block)
            target[base:base + block_size] = (value * scale for value in block)
        transformed.append(target)
    return transformed


def _full_forward(rows: Sequence[Sequence[float]], transpose_u172: bool) -> list[list[float]]:
    matrix = u172()
    scale = 1.0 / math.sqrt(D_FF)
    transformed: list[list[float]] = []
    for source in rows:
        blocks = []
        for k_index in range(FULL_K):
            block = [float(value) for value in source[k_index * FULL_Q:(k_index + 1) * FULL_Q]]
            _fht_in_place(block)
            blocks.append(block)

        output = [[0.0] * FULL_Q for _ in range(FULL_K)]
        for out_k in range(FULL_K):
            accumulator = output[out_k]
            for in_k in range(FULL_K):
                sign = matrix[in_k][out_k] if transpose_u172 else matrix[out_k][in_k]
                source_block = blocks[in_k]
                if sign == 1:
                    for q_index in range(FULL_Q):
                        accumulator[q_index] += source_block[q_index]
                else:
                    for q_index in range(FULL_Q):
                        accumulator[q_index] -= source_block[q_index]
            for q_index in range(FULL_Q):
                accumulator[q_index] *= scale
        transformed.append([value for block in output for value in block])
    return transformed


def forward_rows(rows: Sequence[Sequence[float]], transform_id: str):
    """Apply ``X R_t``. Identity returns the original object without a copy."""

    _validate_rows(rows)
    spec = transform_spec(transform_id)
    if transform_id == "I":
        return rows
    if spec.block_size is not None:
        return _block_forward(rows, spec.block_size)
    return _full_forward(rows, transpose_u172=False)


def inverse_rows(rows: Sequence[Sequence[float]], transform_id: str):
    """Apply ``X R_t^T``. Identity returns the original object without a copy."""

    _validate_rows(rows)
    spec = transform_spec(transform_id)
    if transform_id == "I":
        return rows
    if spec.block_size is not None:
        return _block_forward(rows, spec.block_size)
    return _full_forward(rows, transpose_u172=True)


def fold_stored_weight(weight_rows: Sequence[Sequence[float]], transform_id: str):
    """Fold a stored ``[out_features, in_features]`` weight as ``A R_t``."""

    return forward_rows(weight_rows, transform_id)


def _sylvester_sign(row: int, column: int) -> int:
    return -1 if (row & column).bit_count() & 1 else 1


def direct_forward_coordinate(row: Sequence[float], transform_id: str, output_index: int) -> float:
    """Independent dense definition used to check reshape and fold orientation."""

    if len(row) != D_FF:
        raise ValueError(f"row width must be {D_FF}")
    if not 0 <= output_index < D_FF:
        raise ValueError(f"output index must be in [0, {D_FF})")
    spec = transform_spec(transform_id)
    if transform_id == "I":
        return float(row[output_index])
    if spec.block_size is not None:
        block_size = spec.block_size
        block_base = output_index // block_size * block_size
        block_column = output_index % block_size
        total = sum(float(row[block_base + source]) * _sylvester_sign(source, block_column)
                    for source in range(block_size))
        return total / math.sqrt(block_size)

    out_k, out_q = divmod(output_index, FULL_Q)
    matrix = u172()
    total = 0.0
    for in_k in range(FULL_K):
        k_sign = matrix[out_k][in_k]
        base = in_k * FULL_Q
        for in_q in range(FULL_Q):
            total += k_sign * _sylvester_sign(in_q, out_q) * float(row[base + in_q])
    return total / math.sqrt(D_FF)


def linear(rows: Sequence[Sequence[float]], weight_rows: Sequence[Sequence[float]], bias: Sequence[float]):
    """Small dependency-free ``X A^T + b`` helper for the bounded oracle."""

    _validate_rows(rows)
    _validate_rows(weight_rows)
    if len(weight_rows) != len(bias):
        raise ValueError("bias length must match stored weight rows")
    return [[sum(float(lhs) * float(rhs) for lhs, rhs in zip(row, weight)) + float(offset)
             for weight, offset in zip(weight_rows, bias)] for row in rows]
