"""CPU-only correctness oracle for the Phase A Hurwitz decoder."""

from .oracle import (
    PRIMARY_COUNT,
    build_joint_table,
    decode_factorized,
    decode_joint,
    decode_specialized,
    decode_variants,
    hamilton,
    primary_units,
    split_ids,
)

__all__ = [name for name in globals() if not name.startswith("_")]
