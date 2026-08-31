"""Deterministic dynamic per-row signed-A4 callback for Phase A.

The callback intentionally stops at a dequantized activation tensor.  It is a
one-launch activation-quantization boundary compatible with a later W4A4
linear path; it is not an INT4 GEMM, weight conversion, packer, or fusion.
"""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Sequence

from .reference import D_FF
from .triton_transform import _tensors_overlap

A4_QMIN = -7
A4_QMAX = 7


def quantize_row_reference(row: Sequence[float]) -> tuple[list[float], float]:
    """Dependency-free round-to-nearest-even signed-A4 fake quantization."""

    if len(row) != D_FF:
        raise ValueError(f"activation row width must be exactly {D_FF}")
    values = [float(value) for value in row]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("activation values must be finite")
    absmax = max(abs(value) for value in values)
    scale = absmax / A4_QMAX if absmax else 1.0
    quantized = [max(A4_QMIN, min(A4_QMAX, round(value / scale))) * scale for value in values]
    return quantized, scale


def quantize_rows_reference(rows: Sequence[Sequence[float]]) -> tuple[list[list[float]], list[float]]:
    if not rows:
        raise ValueError("at least one activation row is required")
    outputs = []
    scales = []
    for row in rows:
        output, scale = quantize_row_reference(row)
        outputs.append(output)
        scales.append(scale)
    return outputs, scales


@lru_cache(maxsize=1)
def _kernel_bundle():
    import triton
    import triton.language as tl

    @triton.jit
    def quantize_dequantize_a4_kernel(source, output, scales, WIDTH: tl.constexpr, BLOCK: tl.constexpr,
                                      QMIN: tl.constexpr, QMAX: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < WIDTH
        values = tl.load(source + row * WIDTH + offsets, mask=mask, other=0.0).to(tl.float32)
        absmax = tl.max(tl.abs(values), axis=0)
        scale = tl.where(absmax > 0.0, absmax / QMAX, 1.0)
        rounded = tl.extra.libdevice.rint(values / scale)
        quantized = tl.maximum(QMIN, tl.minimum(QMAX, rounded))
        tl.store(output + row * WIDTH + offsets, quantized * scale, mask=mask)
        tl.store(scales + row, scale)

    return quantize_dequantize_a4_kernel


class W4A4ActivationQuantizer:
    """Prepared one-launch dynamic per-row signed-A4 callback."""

    launches_per_call = 1
    copies_per_call = 0
    qmin = A4_QMIN
    qmax = A4_QMAX
    rounding = "nearest-even"
    granularity = "dynamic-per-row"

    def __init__(self, example):
        import torch

        self._validate(example)
        self.shape = tuple(example.shape)
        self.dtype = example.dtype
        self.device = example.device
        self.output = torch.empty_like(example)
        self.scales = torch.empty(example.shape[0], dtype=torch.float32, device=example.device)

    def _validate(self, tensor) -> None:
        if getattr(tensor, "ndim", None) != 2 or tensor.shape[0] <= 0 or tensor.shape[1] != D_FF:
            raise ValueError(f"A4 quantization expects a rank-2 [N, {D_FF}] tensor")
        if not getattr(tensor, "is_cuda", False):
            raise ValueError("A4 profiling requires a CUDA tensor")
        if str(getattr(tensor, "dtype", None)) != "torch.float16":
            raise ValueError("Phase A A4 profiling requires torch.float16")
        if not tensor.is_contiguous():
            raise ValueError("A4 profiling refuses hidden contiguous copies")

    def __call__(self, tensor):
        self._validate(tensor)
        if tuple(tensor.shape) != self.shape or tensor.dtype != self.dtype or tensor.device != self.device:
            raise ValueError("input must match the prepared quantizer shape, dtype, and device")
        if _tensors_overlap(tensor, self.output):
            raise ValueError("A4 quantization requires distinct input and output buffers")
        kernel = _kernel_bundle()
        kernel[(self.shape[0], )](tensor, self.output, self.scales, WIDTH=D_FF, BLOCK=16384,
                                 QMIN=A4_QMIN, QMAX=A4_QMAX, num_warps=8)
        return self.output
