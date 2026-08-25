"""Pinned QuaRot order-172 Hadamard matrix.

The packed matrix data is adapted from ``fake_quant/hadamard_utils.py`` in
``kw7243/QuaRot@5008669b08c1f11f9b64d52d16fddd47ca754c5a`` (Apache-2.0).
See ``THIRD_PARTY_NOTICES.md`` in this directory. A set bit encodes +1 and a
clear bit encodes -1, in row-major order.
"""

from __future__ import annotations

import base64
import hashlib
from functools import lru_cache

ORDER = 172
PACKED_SHA256 = "891685e890ac3afdc2b0b9515edd890ef601e2319f065182091bc2f4e2a3c005"
INT8_ROW_MAJOR_SHA256 = "378ef12c7cc31f3ea558e1c66b9552a83128f8f732b0bc094b864c8e120d84bd"
MATRIX_DIGEST = f"sha256:int8-row-major:{INT8_ROW_MAJOR_SHA256}"

_PACKED_BASE64 = """
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
"""


def _packed_bytes() -> bytes:
    packed = base64.b64decode(_PACKED_BASE64, validate=False)
    if len(packed) != (ORDER * ORDER + 7) // 8:
        raise RuntimeError("pinned U_172 payload has the wrong length")
    if hashlib.sha256(packed).hexdigest() != PACKED_SHA256:
        raise RuntimeError("pinned U_172 packed digest mismatch")
    return packed


@lru_cache(maxsize=1)
def u172() -> tuple[tuple[int, ...], ...]:
    """Return the exact pinned U_172 as immutable signed integer rows."""

    values: list[int] = []
    for byte in _packed_bytes():
        values.extend(1 if byte & (1 << bit) else -1 for bit in range(7, -1, -1))
    del values[ORDER * ORDER:]

    canonical = bytes(value & 0xFF for value in values)
    if hashlib.sha256(canonical).hexdigest() != INT8_ROW_MAJOR_SHA256:
        raise RuntimeError("pinned U_172 matrix digest mismatch")
    return tuple(tuple(values[row * ORDER:(row + 1) * ORDER]) for row in range(ORDER))


def verify_orthogonality() -> None:
    """Fail unless ``U_172 U_172^T == 172 I`` exactly."""

    matrix = u172()
    for lhs in range(ORDER):
        for rhs in range(lhs, ORDER):
            dot = sum(a * b for a, b in zip(matrix[lhs], matrix[rhs]))
            expected = ORDER if lhs == rhs else 0
            if dot != expected:
                raise RuntimeError(f"pinned U_172 is not orthogonal at ({lhs}, {rhs}): {dot}")
