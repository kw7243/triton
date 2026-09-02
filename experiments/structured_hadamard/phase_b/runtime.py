"""Torch runtime extensions for genuine H32/H128 Phase B transforms."""

from __future__ import annotations

import math
from typing import Mapping

from experiments.structured_hadamard.phase_a.real_model import runtime as phase_a_runtime

from .reference import BLOCK_SIZES, TRANSFORMS, block_semantics


class TorchBlockHadamard:
    """Normalized block-diagonal Sylvester Hadamard using ordinary Torch ops."""

    def __init__(self, width: int, block_size: int, torch_module: object):
        if type(width) is not int or width <= 0 or width % block_size:
            raise phase_a_runtime.PreparationError("block Hadamard width is not divisible by block size")
        if type(block_size) is not int or block_size <= 0 or block_size & (block_size - 1):
            raise phase_a_runtime.PreparationError("block size must be a positive power of two")
        self.width = width
        self.block_size = block_size
        self.torch = torch_module
        self.name = f"H{block_size}"

    def _apply(self, tensor: object):
        if tensor.shape[-1] != self.width:
            raise phase_a_runtime.PreparationError(f"block Hadamard input width must be {self.width}")
        torch = self.torch
        original_shape = tensor.shape
        work = tensor.reshape(-1, self.width // self.block_size, self.block_size).float()
        half = 1
        while half < self.block_size:
            grouped = work.reshape(*work.shape[:-1], -1, 2 * half)
            lhs, rhs = grouped[..., :half], grouped[..., half:]
            work = torch.cat((lhs + rhs, lhs - rhs), dim=-1).reshape(work.shape)
            half *= 2
        return (work.reshape(original_shape) / math.sqrt(self.block_size)).to(tensor.dtype)

    def online(self, tensor: object):
        return self._apply(tensor)

    def fold_weight(self, tensor: object):
        return self._apply(tensor)

    def inverse(self, tensor: object):
        return self._apply(tensor)


def make_transform(transform: str, width: int, *, torch_module: object,
                   full_outer_matrix: object | None = None,
                   full_outer_order: int | None = None):
    if transform not in TRANSFORMS:
        raise phase_a_runtime.PreparationError(f"unknown Phase B transform: {transform}")
    if transform == "I":
        return phase_a_runtime.IdentityTransform()
    if transform in ("H32", "H128"):
        block_size = BLOCK_SIZES[transform]
        assert block_size is not None
        return TorchBlockHadamard(width, block_size, torch_module)
    if full_outer_matrix is None or full_outer_order is None:
        raise phase_a_runtime.PreparationError("Hfull requires the pinned QuaRot outer factor")
    return phase_a_runtime.TorchFullHadamard(
        width, full_outer_matrix, full_outer_order, torch_module,
    )


def transform_semantics(transform: str, width: int, *, full_outer_order: int) -> dict[str, object]:
    return block_semantics(transform, width, full_outer_order=full_outer_order)


def replace_all_projection_linears_for_policy(model: object, *, extension: object,
                                              down_transforms: Mapping[int, object],
                                              torch_module: object) -> tuple[str, ...]:
    """Install accepted packed W4A4 with an explicit transform at every down site."""

    if set(down_transforms) != set(range(len(model.model.layers))):
        raise phase_a_runtime.PreparationError("policy must bind all and only model layer indices")
    wrapper_class = phase_a_runtime._packed_linear_class(torch_module)
    replaced = []
    attention_before = tuple(layer.self_attn for layer in model.model.layers)
    for layer_index, layer in enumerate(model.model.layers):
        for parent_name, child_name in phase_a_runtime.LLAMA_PROJECTION_PATHS:
            parent = getattr(layer, parent_name)
            original = getattr(parent, child_name)
            transform = (
                down_transforms[layer_index]
                if (parent_name, child_name) == ("mlp", "down_proj")
                else phase_a_runtime.IdentityTransform()
            )
            setattr(parent, child_name, wrapper_class(original, extension, transform))
            replaced.append(f"model.layers.{layer_index}.{parent_name}.{child_name}")
    attention_after = tuple(layer.self_attn for layer in model.model.layers)
    if any(before is not after for before, after in zip(attention_before, attention_after)):
        raise phase_a_runtime.PreparationError("attention container changed while applying Phase B policy")
    if len(replaced) != len(model.model.layers) * len(phase_a_runtime.LLAMA_PROJECTION_PATHS):
        raise phase_a_runtime.PreparationError("packed policy replacement count differs")
    return tuple(replaced)
