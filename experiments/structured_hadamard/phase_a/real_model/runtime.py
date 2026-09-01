"""Honest Llama-3 preparation for the packed signed-W4A4 down projection.

Model loading and CUDA imports stay lazy. CPU helpers specify the exact GQA,
packing, scaling, and full-Hadamard contracts. Runtime helpers load real cached
weights into the standard Transformers Llama implementation and replace only
selected FFN ``down_proj`` modules.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import sys
from types import ModuleType
from typing import Callable, Iterable, Sequence


MODEL_REPOSITORY = "NousResearch/Meta-Llama-3-8B"
MODEL_REVISION = "315b20096dc791d381d514deb5f8bd9c8d6d3061"
MODEL_METADATA_SHA256 = {
    "config.json": "2430cee764b6530ff8673cf9ba8561e1d5a33152d503cd0de909ff5718261441",
    "model.safetensors.index.json": "146776fce3f6db1103aa6f249e65ee5544c5923ce6f971b092eee79aa6e5d37b",
}
MODEL_WEIGHT_BLOBS = {
    "model-00001-of-00004.safetensors":
        (4976698672, "f2c144103072514542e327fa8080bd375cb300f2d453fba9ca3aea81d0d4cf33"),
    "model-00002-of-00004.safetensors":
        (4999802720, "d9eee5f23d94405d90b7e9ff88b9443fee42f8528a658f54214c2aba7530d80c"),
    "model-00003-of-00004.safetensors":
        (4915916176, "4b8fbc5e113f69768dd8de84661ea20af8a32b734a9976144b4236c447b40ccc"),
    "model-00004-of-00004.safetensors":
        (1168138808, "5dc34e6bdf2da9e35f0d93b5c333c870f3677dc43dc3a91ea3a8ad28a1fe1acb"),
}
QUAROT_COMMIT = "5008669b08c1f11f9b64d52d16fddd47ca754c5a"
HADAMARD_RELATIVE_PATH = Path("quarot/functional/hadamard.py")
SIGNED_INT4_MIN = -8
SIGNED_INT4_MAX = 7


class PreparationError(RuntimeError):
    """A real-model or packed-runtime identity is not the accepted one."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_regular(path: Path, *, executable: bool = False) -> Path:
    resolved = path.resolve(strict=True)
    status = resolved.stat()
    if not stat.S_ISREG(status.st_mode):
        raise PreparationError(f"required path is not a regular file: {path}")
    if status.st_uid != os.getuid():
        raise PreparationError(f"required file is not owned by uid {os.getuid()}: {resolved}")
    if executable and not os.access(resolved, os.X_OK):
        raise PreparationError(f"required file is not executable: {resolved}")
    return resolved


@dataclass(frozen=True)
class LlamaDimensions:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int

    @classmethod
    def from_config(cls, config: object) -> "LlamaDimensions":
        values = {
            name: getattr(config, name)
            for name in (
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
            )
        }
        if any(type(value) is not int or value <= 0 for value in values.values()):
            raise PreparationError("Llama dimensions must be positive integers")
        hidden = values["hidden_size"]
        attention_heads = values["num_attention_heads"]
        key_value_heads = values["num_key_value_heads"]
        if hidden % attention_heads:
            raise PreparationError("hidden_size must be divisible by num_attention_heads")
        if attention_heads % key_value_heads:
            raise PreparationError("num_attention_heads must be divisible by num_key_value_heads")
        return cls(**values, head_dim=hidden // attention_heads)

    def cache_shape(self, batch_size: int, sequence_length: int) -> tuple[int, int, int, int]:
        if type(batch_size) is not int or batch_size <= 0:
            raise PreparationError("batch_size must be a positive integer")
        if type(sequence_length) is not int or sequence_length <= 0:
            raise PreparationError("sequence_length must be a positive integer")
        return (batch_size, self.num_key_value_heads, sequence_length, self.head_dim)


def verify_llama3_dimensions(config: object) -> LlamaDimensions:
    dimensions = LlamaDimensions.from_config(config)
    expected = (4096, 14336, 32, 32, 8, 128)
    observed = (
        dimensions.hidden_size,
        dimensions.intermediate_size,
        dimensions.num_hidden_layers,
        dimensions.num_attention_heads,
        dimensions.num_key_value_heads,
        dimensions.head_dim,
    )
    if observed != expected:
        raise PreparationError(f"cached Llama-3 dimensions differ: {observed!r}")
    return dimensions


def verify_cached_snapshot(snapshot: Path, *, hash_weight_blobs: bool = False) -> dict[str, object]:
    snapshot = snapshot.resolve(strict=True)
    if snapshot.name != MODEL_REVISION:
        raise PreparationError(f"snapshot directory must be exact revision {MODEL_REVISION}")
    if not snapshot.is_dir():
        raise PreparationError("snapshot must be a directory")

    metadata = {}
    for name, expected in MODEL_METADATA_SHA256.items():
        path = _require_regular(snapshot / name)
        actual = sha256_file(path)
        if actual != expected:
            raise PreparationError(f"cached metadata digest mismatch for {name}: {actual}")
        metadata[name] = actual

    weights = {}
    for name, (expected_size, expected_sha256) in MODEL_WEIGHT_BLOBS.items():
        path = _require_regular(snapshot / name)
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise PreparationError(f"cached weight size mismatch for {name}: {actual_size}")
        actual_sha256 = sha256_file(path) if hash_weight_blobs else None
        if actual_sha256 is not None and actual_sha256 != expected_sha256:
            raise PreparationError(f"cached weight digest mismatch for {name}: {actual_sha256}")
        weights[name] = {
            "bytes": actual_size,
            "expected_sha256": expected_sha256,
            "sha256_verified": hash_weight_blobs,
        }

    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    dimensions = verify_llama3_dimensions(type("Config", (), config)())
    index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise PreparationError("weight index must contain a nonempty weight_map")
    if set(weight_map.values()) != set(MODEL_WEIGHT_BLOBS):
        raise PreparationError("weight index does not reference the exact four accepted shards")
    return {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "snapshot": str(snapshot),
        "metadata_sha256": metadata,
        "weight_blobs": weights,
        "weight_keys": len(weight_map),
        "dimensions": dimensions.__dict__,
        "attention": "transformers-standard-eager",
    }


def _expected_weight_keys(snapshot: Path) -> frozenset[str]:
    index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
    return frozenset(index["weight_map"])


def verify_loaded_weights(model: object, snapshot: Path) -> None:
    state = model.state_dict()
    expected = _expected_weight_keys(snapshot)
    observed = frozenset(state)
    if observed != expected:
        missing = sorted(expected - observed)[:5]
        unexpected = sorted(observed - expected)[:5]
        raise PreparationError(f"loaded state keys differ; missing={missing}, unexpected={unexpected}")
    meta = [name for name, tensor in state.items() if bool(getattr(tensor, "is_meta", False))]
    if meta:
        raise PreparationError(f"loaded model retains uninitialized meta tensors: {meta[:5]}")


def load_cached_llama3(snapshot: Path, *, transformers_module: ModuleType | object | None = None):
    """Load actual cached tensors into standard Llama attention, never a timing skeleton."""

    snapshot = Path(snapshot).resolve(strict=True)
    verify_cached_snapshot(snapshot, hash_weight_blobs=True)
    if transformers_module is None:
        import torch
        import transformers as transformers_module  # type: ignore[no-redef]
        torch_dtype = torch.float16
    else:
        # A string keeps the bounded CPU/static loader test independent of Torch.
        torch_dtype = "float16"
    model = transformers_module.LlamaForCausalLM.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    verify_llama3_dimensions(model.config)
    verify_loaded_weights(model, snapshot)
    return model


def signed_int4_scale_rows(rows: Sequence[Sequence[float]]) -> list[float]:
    if not rows or any(not row for row in rows):
        raise PreparationError("scale input must be a nonempty rectangular matrix")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise PreparationError("scale input must be rectangular")
    return [max(max(abs(float(value)) for value in row) / SIGNED_INT4_MAX, 1.0e-12) for row in rows]


def quantize_signed_int4_rows(rows: Sequence[Sequence[float]], scales: Sequence[float]) -> list[list[int]]:
    if len(rows) != len(scales):
        raise PreparationError("one scale is required per row")
    result = []
    for row, scale in zip(rows, scales):
        if not math.isfinite(scale) or scale <= 0:
            raise PreparationError("scales must be positive and finite")
        result.append([
            max(SIGNED_INT4_MIN, min(SIGNED_INT4_MAX, round(float(value) / scale))) for value in row
        ])
    return result


def pack_signed_int4_rows(rows: Sequence[Sequence[int]]) -> bytes:
    if not rows:
        raise PreparationError("at least one quantized row is required")
    width = len(rows[0])
    if width == 0 or width % 2 or any(len(row) != width for row in rows):
        raise PreparationError("packed signed-int4 rows must be rectangular with even width")
    packed = bytearray()
    for row in rows:
        for index in range(0, width, 2):
            low, high = row[index], row[index + 1]
            if not SIGNED_INT4_MIN <= low <= SIGNED_INT4_MAX:
                raise PreparationError("signed-int4 value is outside [-8, 7]")
            if not SIGNED_INT4_MIN <= high <= SIGNED_INT4_MAX:
                raise PreparationError("signed-int4 value is outside [-8, 7]")
            packed.append((low & 0xF) | ((high & 0xF) << 4))
    return bytes(packed)


def unpack_signed_int4_rows(packed: bytes, *, rows: int, columns: int) -> list[list[int]]:
    if rows <= 0 or columns <= 0 or columns % 2 or len(packed) != rows * columns // 2:
        raise PreparationError("packed shape is invalid")
    result = []
    offset = 0
    for _ in range(rows):
        row = []
        for _ in range(columns // 2):
            value = packed[offset]
            offset += 1
            low, high = value & 0xF, value >> 4
            row.extend((low - 16 if low >= 8 else low, high - 16 if high >= 8 else high))
        result.append(row)
    return result


def _fht_in_place(values: list[float]) -> None:
    if not values or len(values) & (len(values) - 1):
        raise PreparationError("inner Hadamard width must be a positive power of two")
    half = 1
    while half < len(values):
        for base in range(0, len(values), half * 2):
            for offset in range(half):
                lhs = values[base + offset]
                rhs = values[base + half + offset]
                values[base + offset] = lhs + rhs
                values[base + half + offset] = lhs - rhs
        half *= 2


def factored_hadamard_rows(rows: Sequence[Sequence[float]], outer: Sequence[Sequence[int]],
                           *, transpose_outer: bool = False) -> list[list[float]]:
    """Dependency-free ``H_outer tensor H_inner`` row-transform reference."""

    if not rows or not outer or any(len(row) != len(outer) for row in outer):
        raise PreparationError("outer matrix must be nonempty and square")
    order = len(outer)
    width = len(rows[0])
    if width % order or any(len(row) != width for row in rows):
        raise PreparationError("rows must be rectangular and divisible by the outer order")
    inner = width // order
    if inner <= 0 or inner & (inner - 1):
        raise PreparationError("inner Hadamard width must be a positive power of two")
    if any(value not in (-1, 1) for row in outer for value in row):
        raise PreparationError("outer matrix entries must be signs")
    scale = 1.0 / math.sqrt(width)
    output = []
    for source in rows:
        blocks = []
        for block_index in range(order):
            begin = block_index * inner
            block = [float(value) for value in source[begin:begin + inner]]
            _fht_in_place(block)
            blocks.append(block)
        transformed = [[0.0] * inner for _ in range(order)]
        for out_index in range(order):
            for in_index in range(order):
                sign = outer[in_index][out_index] if transpose_outer else outer[out_index][in_index]
                for inner_index in range(inner):
                    transformed[out_index][inner_index] += sign * blocks[in_index][inner_index]
        output.append([value * scale for block in transformed for value in block])
    return output


def fold_weight_rows(rows: Sequence[Sequence[float]], outer: Sequence[Sequence[int]]) -> list[list[float]]:
    return factored_hadamard_rows(rows, outer)


def _git_head(root: Path) -> str:
    import subprocess

    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=root, check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if completed.returncode:
        raise PreparationError(f"cannot identify dependency Git HEAD: {completed.stderr.strip()}")
    return completed.stdout.strip()


def load_quarot_outer_matrix(quarot_root: Path, width: int, torch_module: ModuleType | object):
    """Read the exact unchanged QuaRot matrix definition without copying it."""

    quarot_root = quarot_root.resolve(strict=True)
    if _git_head(quarot_root) != QUAROT_COMMIT:
        raise PreparationError(f"QuaRot must be exact commit {QUAROT_COMMIT}")
    source = _require_regular(quarot_root / HADAMARD_RELATIVE_PATH)
    previous = sys.modules.get("fast_hadamard_transform")
    sys.modules["fast_hadamard_transform"] = ModuleType("fast_hadamard_transform")
    try:
        spec = importlib.util.spec_from_file_location("_phase_a_pinned_quarot_hadamard", source)
        if spec is None or spec.loader is None:
            raise PreparationError("cannot load pinned QuaRot Hadamard definition")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        matrix, order = module.get_hadK(width)
    finally:
        if previous is None:
            sys.modules.pop("fast_hadamard_transform", None)
        else:
            sys.modules["fast_hadamard_transform"] = previous
    inner = width // order if order else 0
    if order is None or width % order or inner <= 0 or inner & (inner - 1):
        raise PreparationError("QuaRot returned an invalid full-Hadamard factorization")
    if matrix is None:
        matrix = torch_module.ones((1, 1), dtype=torch_module.float32)
    return matrix, order


class TorchFullHadamard:
    """Exact QuaRot factorization implemented with ordinary Torch operations."""

    def __init__(self, width: int, outer_matrix: object, outer_order: int, torch_module: object):
        self.width = width
        self.outer_matrix = outer_matrix
        self.outer_order = outer_order
        self.torch = torch_module

    def _apply(self, tensor: object, *, transpose_outer: bool = False):
        torch = self.torch
        original_shape = tensor.shape
        if original_shape[-1] != self.width:
            raise PreparationError(f"Hadamard input width must be {self.width}")
        inner = self.width // self.outer_order
        work = tensor.reshape(-1, self.outer_order, inner).float()
        half = 1
        while half < inner:
            grouped = work.reshape(*work.shape[:-1], -1, half * 2)
            lhs, rhs = grouped[..., :half], grouped[..., half:]
            work = torch.cat((lhs + rhs, lhs - rhs), dim=-1).reshape(work.shape)
            half *= 2
        matrix = self.outer_matrix.to(device=work.device, dtype=work.dtype)
        if transpose_outer:
            matrix = matrix.t()
        work = torch.einsum("oi,bit->bot", matrix, work)
        return (work.reshape(original_shape) / math.sqrt(self.width)).to(tensor.dtype)

    def online(self, tensor: object):
        return self._apply(tensor)

    def fold_weight(self, tensor: object):
        return self._apply(tensor)

    def inverse(self, tensor: object):
        return self._apply(tensor, transpose_outer=True)


def load_extension(path: Path, expected_sha256: str | None = None):
    path = _require_regular(path)
    actual = sha256_file(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise PreparationError(f"extension digest mismatch: {actual}")
    spec = importlib.util.spec_from_file_location("phase_a_w4a4_cuda", path)
    if spec is None or spec.loader is None:
        raise PreparationError("cannot construct extension import specification")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("matmul", "sym_quant", "sym_dequant"):
        if not callable(getattr(module, name, None)):
            raise PreparationError(f"extension lacks callable {name}")
    return module


def _packed_down_projection_class(torch: object):
    class PackedW4A4DownProjection(torch.nn.Module):
        def __init__(self, original: object, extension: object, transform: TorchFullHadamard):
            super().__init__()
            if original.bias is not None:
                raise PreparationError("Llama down_proj is expected to have no bias")
            self.in_features = original.in_features
            self.out_features = original.out_features
            self.extension = extension
            self.transform = transform
            folded = transform.fold_weight(original.weight.detach())
            scales = (folded.float().abs().amax(dim=1) / SIGNED_INT4_MAX).clamp_min(1.0e-12)
            scales = scales.to(folded.dtype).contiguous()
            packed = extension.sym_quant(folded.contiguous(), scales)
            self.register_buffer("weight", packed)
            self.register_buffer("weight_scales", scales)

        def forward(self, inputs: object):
            original_shape = inputs.shape
            flattened = inputs.reshape(-1, self.in_features)
            transformed = self.transform.online(flattened)
            scales = (transformed.float().abs().amax(dim=1) / SIGNED_INT4_MAX).clamp_min(1.0e-12)
            scales = scales.to(transformed.dtype).contiguous()
            packed = self.extension.sym_quant(transformed.contiguous(), scales)
            accumulated = self.extension.matmul(packed, self.weight)
            output = self.extension.sym_dequant(accumulated, scales, self.weight_scales)
            return output.reshape(*original_shape[:-1], self.out_features)

    return PackedW4A4DownProjection


def replace_down_projections(model: object, *, layers: Iterable[int], extension: object,
                             transform_factory: Callable[[int], object], torch_module: object) -> tuple[int, ...]:
    """Replace only selected ``mlp.down_proj`` modules; attention is untouched."""

    selected = tuple(layers)
    if not selected or len(set(selected)) != len(selected):
        raise PreparationError("layers must be nonempty and unique")
    model_layers = model.model.layers
    if any(type(index) is not int or not 0 <= index < len(model_layers) for index in selected):
        raise PreparationError("selected layer index is outside the model")
    attention_before = tuple(layer.self_attn for layer in model_layers)
    wrapper_class = _packed_down_projection_class(torch_module)
    for index in selected:
        layer = model_layers[index]
        original = layer.mlp.down_proj
        transform = transform_factory(original.in_features)
        layer.mlp.down_proj = wrapper_class(original, extension, transform)
    attention_after = tuple(layer.self_attn for layer in model_layers)
    if any(before is not after for before, after in zip(attention_before, attention_after)):
        raise PreparationError("standard model attention changed during down-projection replacement")
    return selected


def prepare_model(snapshot: Path, extension_path: Path, extension_sha256: str, quarot_root: Path,
                  *, layers: Iterable[int] = (0,), device: str = "cuda:0"):
    import torch

    model = load_cached_llama3(snapshot).to(device)
    extension = load_extension(extension_path, extension_sha256)

    def transform_factory(width: int) -> TorchFullHadamard:
        outer, order = load_quarot_outer_matrix(quarot_root, width, torch)
        return TorchFullHadamard(width, outer, order, torch)

    replace_down_projections(
        model, layers=layers, extension=extension, transform_factory=transform_factory, torch_module=torch,
    )
    return model
