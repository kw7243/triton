"""Runnable Triton boundary for Phase A ``I`` and exact ``Hfull``.

Imports of Torch and Triton are deliberately lazy. The identity branch returns
before either dependency is imported and never launches or copies. ``Hfull``
uses the exact QuaRot-compatible ``172 x 64`` factorization: a natural-order
64-point FHT followed by left multiplication with the pinned ``U_172`` and one
``1/sqrt(11008)`` normalization.
"""

from __future__ import annotations

from functools import lru_cache

from .reference import D_FF, FULL_K, FULL_Q
from .u172 import u172


@lru_cache(maxsize=1)
def _kernel_bundle():
    import triton
    import triton.language as tl

    @triton.jit
    def fht64_kernel(source, intermediate):
        row_k = tl.program_id(0)
        offsets = tl.arange(0, 64)
        values = tl.load(source + row_k * 64 + offsets)

        values = values.reshape(32, 2)
        lhs, rhs = tl.split(values)
        values = tl.join(lhs + rhs, lhs - rhs).reshape(64)

        values = values.reshape(16, 2, 2).permute(0, 2, 1)
        lhs, rhs = tl.split(values)
        values = tl.join(lhs + rhs, lhs - rhs).permute(0, 2, 1).reshape(64)

        values = values.reshape(8, 2, 4).permute(0, 2, 1)
        lhs, rhs = tl.split(values)
        values = tl.join(lhs + rhs, lhs - rhs).permute(0, 2, 1).reshape(64)

        values = values.reshape(4, 2, 8).permute(0, 2, 1)
        lhs, rhs = tl.split(values)
        values = tl.join(lhs + rhs, lhs - rhs).permute(0, 2, 1).reshape(64)

        values = values.reshape(2, 2, 16).permute(0, 2, 1)
        lhs, rhs = tl.split(values)
        values = tl.join(lhs + rhs, lhs - rhs).permute(0, 2, 1).reshape(64)

        values = values.reshape(1, 2, 32).permute(0, 2, 1)
        lhs, rhs = tl.split(values)
        values = tl.join(lhs + rhs, lhs - rhs).permute(0, 2, 1).reshape(64)

        tl.store(intermediate + row_k * 64 + offsets, values)

    @triton.jit
    def u172_kernel(intermediate, matrix, output, INV_SQRT_D: tl.constexpr, BLOCK_OUT: tl.constexpr,
                    BLOCK_IN: tl.constexpr):
        row = tl.program_id(0)
        q_index = tl.program_id(1)
        out_block = tl.program_id(2)

        out_k = out_block * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
        in_k = tl.arange(0, BLOCK_IN)
        values = tl.load(intermediate + row * 11008 + in_k * 64 + q_index, mask=in_k < 172, other=0.0)
        signs = tl.load(matrix + out_k[:, None] * 172 + in_k[None, :],
                        mask=(out_k[:, None] < 172) & (in_k[None, :] < 172), other=0.0)
        accumulated = tl.sum(signs.to(tl.float32) * values[None, :].to(tl.float32), axis=1)
        tl.store(output + row * 11008 + out_k * 64 + q_index, accumulated * INV_SQRT_D, mask=out_k < 172)

    return triton, fht64_kernel, u172_kernel


class HFullWorkspace:
    """Prepared buffers and matrix for repeated ``[N, 11008]`` launches."""

    transform_launches = 2
    transform_copies = 0

    def __init__(self, example):
        import torch

        self._validate(example)
        self.shape = tuple(example.shape)
        self.dtype = example.dtype
        self.device = example.device
        self.intermediate = torch.empty_like(example)
        self.output = torch.empty_like(example)
        # Matrix preparation is outside the timed transform boundary.
        self.matrix = torch.tensor(u172(), dtype=example.dtype, device=example.device).contiguous()

    def _validate(self, tensor) -> None:
        if getattr(tensor, "ndim", None) != 2 or tensor.shape[0] <= 0 or tensor.shape[1] != D_FF:
            raise ValueError(f"Hfull expects a rank-2 [N, {D_FF}] tensor")
        if not getattr(tensor, "is_cuda", False):
            raise ValueError("Hfull profiling requires a CUDA tensor")
        if str(getattr(tensor, "dtype", None)) != "torch.float16":
            raise ValueError("Phase A Hfull profiling requires torch.float16")
        if not tensor.is_contiguous():
            raise ValueError("Hfull refuses hidden contiguous copies")

    def __call__(self, tensor, *, out=None):
        self._validate(tensor)
        if tuple(tensor.shape) != self.shape or tensor.dtype != self.dtype or tensor.device != self.device:
            raise ValueError("input must match the workspace shape, dtype, and device")
        if out is None:
            out = self.output
        elif (tuple(out.shape) != self.shape or out.dtype != self.dtype or out.device != self.device
              or not out.is_contiguous()):
            raise ValueError("output must match the workspace and be contiguous")
        if out.data_ptr() == tensor.data_ptr():
            raise ValueError("Hfull requires distinct input and output buffers")

        triton, fht64_kernel, u172_kernel = _kernel_bundle()
        fht64_kernel[(self.shape[0] * FULL_K, )](tensor, self.intermediate, num_warps=1)
        grid = (self.shape[0], FULL_Q, triton.cdiv(FULL_K, 16))
        u172_kernel[grid](self.intermediate, self.matrix, out, INV_SQRT_D=D_FF**-0.5, BLOCK_OUT=16,
                          BLOCK_IN=256, num_warps=4)
        return out


def apply_transform(tensor, transform_id: str, *, workspace: HFullWorkspace | None = None, out=None):
    """Apply a Phase A transform; ``I`` is a host alias/no-launch branch."""

    if transform_id == "I":
        if out is not None and out is not tensor:
            raise ValueError("identity cannot write a distinct output because that would be hidden copy work")
        return tensor
    if transform_id != "Hfull":
        raise ValueError("the Phase A Triton boundary is runnable only for I and Hfull")
    if workspace is None:
        workspace = HFullWorkspace(tensor)
    return workspace(tensor, out=out)
