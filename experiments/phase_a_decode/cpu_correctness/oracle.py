"""Independent CPU oracle for Phase A J/F/H Hurwitz decoding.

Quaternions are scalar-first ``(w, x, y, z)``. A flat code is decoded as
``id = p * S + s``, where ``p`` selects one of the 24 primary Hurwitz units
and ``s`` selects a row of the secondary table.

Every public operation rejects non-CPU tensors. This module is a no-GPU
correctness lane, not a fallback execution path for the Triton benchmark.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

PRIMARY_COUNT = 24
QUATERNION_WIDTH = 4

# Left multiplication by +1, -1, +i, -i, +j, -j, +k, -k.
_AXIS_INDEX = torch.tensor(
    [
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [2, 3, 0, 1],
        [3, 2, 1, 0],
        [3, 2, 1, 0],
    ],
    dtype=torch.int64,
)
_AXIS_SIGN = torch.tensor(
    [
        [1, 1, 1, 1],
        [-1, -1, -1, -1],
        [-1, 1, -1, 1],
        [1, -1, 1, -1],
        [-1, 1, 1, -1],
        [1, -1, -1, 1],
        [-1, -1, 1, 1],
        [1, 1, -1, -1],
    ],
    dtype=torch.float32,
)


def _require_cpu(name: str, tensor: torch.Tensor) -> None:
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor, got {tensor.device}")


def _require_quaternions(name: str, tensor: torch.Tensor) -> None:
    _require_cpu(name, tensor)
    if tensor.ndim == 0 or tensor.shape[-1] != QUATERNION_WIDTH:
        raise ValueError(f"{name} must have final dimension 4, got {tuple(tensor.shape)}")


def primary_units(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return all 24 primary Hurwitz units in decoder index order."""

    axis = [
        (1, 0, 0, 0),
        (-1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, -1, 0, 0),
        (0, 0, 1, 0),
        (0, 0, -1, 0),
        (0, 0, 0, 1),
        (0, 0, 0, -1),
    ]
    half = [
        tuple(-0.5 if bits & (1 << component) else 0.5 for component in range(4))
        for bits in range(16)
    ]
    return torch.tensor(axis + half, dtype=dtype, device="cpu")


def hamilton(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Compute a scalar-first Hamilton product in float32 on CPU."""

    _require_quaternions("left", left)
    _require_quaternions("right", right)
    aw, ax, ay, az = left.to(torch.float32).unbind(-1)
    bw, bx, by, bz = right.to(torch.float32).unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def split_ids(ids: torch.Tensor, secondary_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split flat ids into primary and secondary indices."""

    _require_cpu("ids", ids)
    if ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError(f"ids must have an integer dtype, got {ids.dtype}")
    if secondary_size <= 0:
        raise ValueError(f"secondary_size must be positive, got {secondary_size}")
    ids64 = ids.to(torch.int64)
    upper = PRIMARY_COUNT * secondary_size
    if ids64.numel() and bool(((ids64 < 0) | (ids64 >= upper)).any()):
        raise IndexError(f"ids must be in [0, {upper})")
    return torch.div(ids64, secondary_size, rounding_mode="floor"), ids64 % secondary_size


def _secondary_size(secondary: torch.Tensor) -> int:
    _require_quaternions("secondary", secondary)
    if secondary.ndim != 2:
        raise ValueError(f"secondary must have shape (S, 4), got {tuple(secondary.shape)}")
    if secondary.shape[0] <= 0:
        raise ValueError("secondary table must not be empty")
    return secondary.shape[0]


def build_joint_table(
    secondary: torch.Tensor, storage_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Materialize J's (24*S, 4) table from stored secondary rows."""

    secondary_size = _secondary_size(secondary)
    stored = secondary.to(storage_dtype)
    joint32 = hamilton(
        primary_units().view(PRIMARY_COUNT, 1, QUATERNION_WIDTH),
        stored.to(torch.float32).view(1, secondary_size, QUATERNION_WIDTH),
    )
    return joint32.reshape(PRIMARY_COUNT * secondary_size, QUATERNION_WIDTH).to(storage_dtype)


def decode_joint(ids: torch.Tensor, joint: torch.Tensor, secondary_size: int) -> torch.Tensor:
    """J: gather a materialized joint table."""

    _require_quaternions("joint", joint)
    if joint.ndim != 2 or joint.shape[0] != PRIMARY_COUNT * secondary_size:
        raise ValueError(
            f"joint must have shape ({PRIMARY_COUNT * secondary_size}, 4), "
            f"got {tuple(joint.shape)}"
        )
    split_ids(ids, secondary_size)
    return joint[ids.to(torch.int64)]


def decode_factorized(ids: torch.Tensor, secondary: torch.Tensor) -> torch.Tensor:
    """F: gather primary and secondary factors, then multiply generically."""

    secondary_size = _secondary_size(secondary)
    p, s = split_ids(ids, secondary_size)
    return hamilton(primary_units()[p], secondary[s].to(torch.float32))


def decode_specialized(ids: torch.Tensor, secondary: torch.Tensor) -> torch.Tensor:
    """H: use signed permutations for axis units and signed sums for half units."""

    secondary_size = _secondary_size(secondary)
    p, s = split_ids(ids, secondary_size)
    values = secondary[s].to(torch.float32)
    flat_p = p.reshape(-1)
    flat_values = values.reshape(-1, QUATERNION_WIDTH)
    flat_out = torch.empty_like(flat_values)

    axis_mask = flat_p < 8
    if bool(axis_mask.any()):
        axis_p = flat_p[axis_mask]
        axis_values = flat_values[axis_mask]
        flat_out[axis_mask] = (
            torch.gather(axis_values, 1, _AXIS_INDEX[axis_p]) * _AXIS_SIGN[axis_p]
        )

    half_mask = ~axis_mask
    if bool(half_mask.any()):
        half_index = flat_p[half_mask] - 8
        half_values = flat_values[half_mask]
        bits = torch.arange(4, dtype=torch.int64)
        signs = torch.where(
            (half_index[:, None] & (1 << bits)) == 0,
            torch.tensor(1.0),
            torch.tensor(-1.0),
        )
        sw, sx, sy, sz = signs.unbind(-1)
        w, x, y, z = half_values.unbind(-1)
        flat_out[half_mask] = 0.5 * torch.stack(
            (
                sw * w - sx * x - sy * y - sz * z,
                sw * x + sx * w + sy * z - sz * y,
                sw * y - sx * z + sy * w + sz * x,
                sw * z + sx * y - sy * x + sz * w,
            ),
            dim=-1,
        )
    return flat_out.reshape(values.shape)


def decode_variants(
    ids: torch.Tensor,
    secondary: torch.Tensor,
    storage_dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Run J/F/H with the benchmark's table-storage round trip."""

    _secondary_size(secondary)
    stored = secondary.to(storage_dtype)
    joint = build_joint_table(stored, storage_dtype)
    return {
        "J": decode_joint(ids, joint, stored.shape[0]),
        "F": decode_factorized(ids, stored).to(storage_dtype),
        "H": decode_specialized(ids, stored).to(storage_dtype),
    }
