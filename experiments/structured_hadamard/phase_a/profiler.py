"""Transform-only and sequential transform-plus-quantize timing boundaries.

This is a library boundary, not a scheduler or experiment CLI. Importing it is
CPU-safe. A caller must already own a prepared CUDA tensor and, for the future
combined identity, a quantization callable. Phase A records ``fusion=none``:
the combined boundary calls the transform and then the quantizer, and makes no
fused-kernel claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .triton_transform import HFullWorkspace, apply_transform

TIMING_IDENTITIES = ("transform-only", "transform+quantize")


@dataclass(frozen=True)
class ProfileSummary:
    timing_identity: str
    p10_us: float
    median_us: float
    p90_us: float
    samples_us: tuple[float, ...]
    transform_launches: int
    transform_copies: int


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    location = quantile * (len(sorted_values) - 1)
    lower = int(location)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = location - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _summarize(samples_ms, timing_identity: str, launches: int, copies: int) -> ProfileSummary:
    samples_us = sorted(float(value) * 1000.0 for value in samples_ms)
    if not samples_us:
        raise RuntimeError("profiler returned no samples")
    return ProfileSummary(timing_identity, _percentile(samples_us, 0.1), _percentile(samples_us, 0.5),
                          _percentile(samples_us, 0.9), tuple(samples_us), launches, copies)


def profile_transform(tensor, transform_id: str, timing_identity: str, *, quantize: Callable | None = None,
                      workspace: HFullWorkspace | None = None, warmup_ms: int = 25,
                      repetition_ms: int = 200) -> ProfileSummary:
    """Profile exactly one declared boundary using Triton's device-event timer."""

    if timing_identity not in TIMING_IDENTITIES:
        raise ValueError(f"timing_identity must be one of {TIMING_IDENTITIES}")
    if timing_identity == "transform+quantize" and quantize is None:
        raise ValueError("transform+quantize requires an explicit quantize callable")
    if warmup_ms <= 0 or repetition_ms <= 0:
        raise ValueError("warmup_ms and repetition_ms must be positive")

    # Identity transform-only is an algebraic zero, not a timed copy or launch.
    if transform_id == "I" and timing_identity == "transform-only":
        return ProfileSummary(timing_identity, 0.0, 0.0, 0.0, (), 0, 0)

    if transform_id == "Hfull" and workspace is None:
        workspace = HFullWorkspace(tensor)

    if timing_identity == "transform-only":
        measured = lambda: apply_transform(tensor, transform_id, workspace=workspace)
    elif transform_id == "I":
        measured = lambda: quantize(tensor)
    else:
        # This is deliberately sequential. Do not describe this boundary as a
        # fused transform/quantization kernel in records or documentation.
        measured = lambda: quantize(apply_transform(tensor, transform_id, workspace=workspace))

    from triton.testing import do_bench

    samples_ms = do_bench(measured, warmup=warmup_ms, rep=repetition_ms, return_mode="all")
    launches = 0 if transform_id == "I" else HFullWorkspace.transform_launches
    return _summarize(samples_ms, timing_identity, launches, 0)
