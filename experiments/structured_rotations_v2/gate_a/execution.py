#!/usr/bin/env python3
"""Structured Rotations v2 Gate A, Experiment A2.

The native consumer is the preserved QuaRot CUTLASS signed-int4 GEMM.  The
frozen A1 asymmetric activation alphabet is mapped exactly onto that signed
consumer by packing ``q_u - 8`` and applying the algebraic zero-point
correction before dequantization.  This file contains no bridge or selector.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from .stage_manifest import verify as verify_stage_manifest


SCHEMA = "structured-rotations-v2-gate-a-execution-v1"
CONTRACT_SCHEMA = "structured-rotations-v2-gate-a-execution-contract-v1"
WIDTH = 12288
HIDDEN = 4096
OUTPUT = 4096
SEGMENT_METHODS = (
    "identity",
    "full_hadamard",
    "local_h32",
    "local_h128",
    "perq_h32",
    "perq_h128",
)
E2E_METHODS = ("identity", "full_hadamard", "local_h32", "local_h128")


class ExperimentError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: object) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def write_json(path: Path, value: object) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write((json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def write_jsonl(path: Path, values: Iterable[object]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        for value in values:
            stream.write((json.dumps(value, sort_keys=True) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def append_event(path: Path, event: str, **values: object) -> None:
    payload = {"recorded_at_utc": utc_now(), "event": event, **values}
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
    with os.fdopen(descriptor, "ab") as stream:
        stream.write((json.dumps(payload, sort_keys=True) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print("EVENT " + json.dumps(payload, sort_keys=True), flush=True)


def run_text(argv: Sequence[str], *, cwd: Path | None = None) -> str:
    return subprocess.check_output(argv, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ExperimentError("cannot summarize an empty timing sample")
    index = fraction * (len(ordered) - 1)
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - index) + ordered[high] * (index - low)


def summarize(values: Sequence[float]) -> dict[str, object]:
    median = statistics.median(values)
    return {
        "samples": len(values),
        "minimum_ms": min(values),
        "p10_ms": percentile(values, 0.10),
        "median_ms": median,
        "p90_ms": percentile(values, 0.90),
        "maximum_ms": max(values),
        "mean_ms": statistics.fmean(values),
        "median_absolute_deviation_ms": statistics.median(
            abs(value - median) for value in values
        ),
    }


def validate_contract(contract: Mapping[str, Any], quality: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise ExperimentError("A2 contract schema differs")
    if contract["scope"]["experiment"] != "A2":
        raise ExperimentError("execution scope differs")
    if tuple(contract["methods"]["segment"]) != SEGMENT_METHODS:
        raise ExperimentError("segment method set differs")
    if tuple(contract["methods"]["end_to_end"]) != E2E_METHODS:
        raise ExperimentError("end-to-end method set differs")
    if contract["budget"]["attempt_limit"] != 1 or contract["budget"]["requeue"]:
        raise ExperimentError("one-shot/no-requeue contract differs")
    if quality["model"]["substitution"] != contract["model"]["repository"]:
        raise ExperimentError("A1/A2 model differs")
    if quality["model"]["revision"] != contract["model"]["revision"]:
        raise ExperimentError("A1/A2 model revision differs")
    if quality["model"]["ffn_intermediate_size"] != WIDTH:
        raise ExperimentError("frozen FFN width differs")
    if quality["quantization"]["activation"] != (
        "asymmetric unsigned INT4 [0,15], dynamic per token row"
    ):
        raise ExperimentError("A1 activation quantizer differs")
    if quality["quantization"]["weight"] != (
        "symmetric signed narrow INT4 [-7,7], per output channel"
    ):
        raise ExperimentError("A1 weight quantizer differs")


def verify_stage(root: Path, contract_path: Path, quality_path: Path) -> dict[str, object]:
    expected_parent = Path(
        json.loads(contract_path.read_text(encoding="utf-8"))["storage"]["staging_parent"]
    ).resolve()
    if root.parent != expected_parent or not (root / ".git").is_dir():
        raise ExperimentError("execution is not in a direct self-contained timestamped stage")
    metadata = root / "REPRODUCIBILITY_METADATA.json"
    if not metadata.is_file():
        raise ExperimentError("research-reproducibility metadata is absent")
    run_text(["git", "fsck", "--connectivity-only", "--no-progress"], cwd=root)
    manifest = verify_stage_manifest(root)
    return {
        "root": str(root),
        "head": run_text(["git", "rev-parse", "HEAD"], cwd=root),
        "tree": run_text(["git", "rev-parse", "HEAD^{tree}"], cwd=root),
        "metadata_sha256": sha256_file(metadata),
        "file_manifest": manifest,
        "execution_contract_sha256": sha256_file(contract_path),
        "quality_contract_stage_sha256": sha256_file(quality_path),
        "quality_contract_source_sha256": (
            "ed49c72bf9dba32e8d154e5205ddeaba2449c83363068c6b05d40a2a3b010878"
        ),
    }


def paley_hadamard_12_rows() -> list[list[int]]:
    prime = 11
    squares = {value * value % prime for value in range(1, prime)}

    def character(value: int) -> int:
        value %= prime
        if value == 0:
            return 0
        return 1 if value in squares else -1

    rows = [[1] * 12]
    for row in range(prime):
        rows.append([-1] + [1 if row == column else character(row - column)
                            for column in range(prime)])
    return rows


def _kernel_bundle():
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def hadamard_tile_kernel(source, output, STAGES: tl.constexpr, INV_SQRT: tl.constexpr):
        tile = tl.program_id(0)
        offsets = tl.arange(0, 1024)
        values = tl.load(source + tile * 1024 + offsets).to(tl.float32)
        for stage in tl.static_range(0, STAGES):
            half: tl.constexpr = 1 << stage
            groups = values.reshape(1024 // (2 * half), 2, half).permute(0, 2, 1)
            lhs, rhs = tl.split(groups)
            values = tl.join(lhs + rhs, lhs - rhs).permute(0, 2, 1).reshape(1024)
        tl.store(output + tile * 1024 + offsets, values * INV_SQRT)

    @triton.jit
    def outer_h12_kernel(intermediate, matrix, output, INV_SQRT: tl.constexpr,
                         BLOCK_K: tl.constexpr):
        row = tl.program_id(0)
        k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
        b = tl.arange(0, 16)[:, None]
        accumulated = tl.zeros((16, BLOCK_K), tl.float32)
        for a in tl.static_range(0, 12):
            values = tl.load(
                intermediate + row * 12288 + a * 1024 + k,
                mask=k < 1024,
                other=0.0,
            ).to(tl.float32)
            signs = tl.load(matrix + a * 16 + b, mask=b < 12, other=0.0).to(tl.float32)
            accumulated += signs * values[None, :]
        offsets = row * 12288 + b * 1024 + k[None, :]
        tl.store(output + offsets, accumulated * INV_SQRT, mask=(b < 12) & (k[None, :] < 1024))

    @triton.jit
    def quantize_pack_kernel(source, scale, zero_point, packed, columns: tl.constexpr,
                             BLOCK_PACKED: tl.constexpr):
        row = tl.program_id(0)
        packed_offsets = tl.program_id(1) * BLOCK_PACKED + tl.arange(0, BLOCK_PACKED)
        mask = packed_offsets < columns // 2
        first = 2 * packed_offsets
        second = first + 1
        row_base = row * columns
        row_scale = tl.load(scale + row).to(tl.float32)
        row_zp = tl.load(zero_point + row).to(tl.float32)
        lhs = tl.load(source + row_base + first, mask=mask, other=0.0).to(tl.float32)
        rhs = tl.load(source + row_base + second, mask=mask, other=0.0).to(tl.float32)
        lhs_q = libdevice.rint(lhs / row_scale + row_zp)
        rhs_q = libdevice.rint(rhs / row_scale + row_zp)
        lhs_q = tl.maximum(0.0, tl.minimum(15.0, lhs_q)).to(tl.int32) - 8
        rhs_q = tl.maximum(0.0, tl.minimum(15.0, rhs_q)).to(tl.int32) - 8
        encoded = (lhs_q & 15) | ((rhs_q & 15) << 4)
        tl.store(packed + row * (columns // 2) + packed_offsets, encoded, mask=mask)

    @triton.jit
    def correction_dequant_kernel(accumulated, activation_scale, zero_point, weight_sum,
                                   weight_scale, output, columns: tl.constexpr,
                                   BLOCK_N: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < columns
        values = tl.load(accumulated + row * columns + offsets, mask=mask, other=0).to(tl.int32)
        zp = tl.load(zero_point + row).to(tl.int32)
        sums = tl.load(weight_sum + offsets, mask=mask, other=0).to(tl.int32)
        corrected = values + (8 - zp) * sums
        a_scale = tl.load(activation_scale + row).to(tl.float32)
        w_scale = tl.load(weight_scale + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(output + row * columns + offsets,
                 corrected.to(tl.float32) * a_scale * w_scale, mask=mask)

    return (
        triton,
        hadamard_tile_kernel,
        outer_h12_kernel,
        quantize_pack_kernel,
        correction_dequant_kernel,
    )


class Transform:
    def __init__(self, method: str, torch: Any):
        if method.startswith("perq_"):
            method = method.removeprefix("perq_").replace("h", "local_h")
        if method not in ("identity", "full_hadamard", "local_h32", "local_h128"):
            raise ExperimentError(f"unsupported transform {method}")
        self.method = method
        self.torch = torch
        padded = [row + [0] * 4 for row in paley_hadamard_12_rows()] + [[0] * 16] * 4
        self.matrix = torch.tensor(padded, device="cuda:0", dtype=torch.float32).contiguous()

    def __call__(self, tensor: Any):
        if tensor.ndim != 2 or int(tensor.shape[1]) != WIDTH or not tensor.is_contiguous():
            raise ExperimentError("transform input must be contiguous [M,12288]")
        if self.method == "identity":
            return tensor
        triton, tile_kernel, outer_kernel, _, _ = _kernel_bundle()
        rows = int(tensor.shape[0])
        if self.method == "full_hadamard":
            intermediate = self.torch.empty((rows, WIDTH), device=tensor.device,
                                            dtype=self.torch.float32)
            output = self.torch.empty_like(intermediate)
            tile_kernel[(rows * 12,)](
                tensor, intermediate, STAGES=10, INV_SQRT=1.0, num_warps=8
            )
            outer_kernel[(rows, triton.cdiv(1024, 64))](
                intermediate,
                self.matrix,
                output,
                INV_SQRT=WIDTH**-0.5,
                BLOCK_K=64,
                num_warps=4,
            )
            return output
        stages = 5 if self.method == "local_h32" else 7
        output = self.torch.empty((rows, WIDTH), device=tensor.device, dtype=self.torch.float32)
        tile_kernel[(rows * 12,)](
            tensor,
            output,
            STAGES=stages,
            INV_SQRT=(32 if stages == 5 else 128) ** -0.5,
            num_warps=8,
        )
        return output


def torch_full_reference(tensor: Any, torch: Any):
    work = tensor.float().reshape(-1, 12, 1024)
    half = 1
    while half < 1024:
        grouped = work.reshape(*work.shape[:-1], -1, 2 * half)
        lhs = grouped[..., :half]
        rhs = grouped[..., half:]
        work = torch.cat((lhs + rhs, lhs - rhs), dim=-1).reshape(work.shape)
        half *= 2
    matrix = torch.tensor(paley_hadamard_12_rows(), device=tensor.device, dtype=torch.float32)
    return torch.einsum("...ak,ab->...bk", work, matrix).reshape(tensor.shape) / math.sqrt(WIDTH)


def pack_signed(values: Any, torch: Any):
    encoded = values.to(torch.int16) & 15
    return (encoded[:, 0::2] | (encoded[:, 1::2] << 4)).to(torch.uint8).contiguous()


def unpack_signed(packed: Any, columns: int, torch: Any):
    values = torch.empty((int(packed.shape[0]), columns), device=packed.device, dtype=torch.int8)
    values[:, 0::2] = (packed & 15).to(torch.int8)
    values[:, 1::2] = (packed >> 4).to(torch.int8)
    return torch.where(values >= 8, values - 16, values)


def quantize_weight(weight: Any, torch: Any, row_chunk: int = 16):
    if tuple(weight.shape) != (OUTPUT, WIDTH):
        raise ExperimentError(f"down weight shape differs: {tuple(weight.shape)}")
    factors = torch.linspace(1.0, 0.8, 21, device=weight.device, dtype=torch.float32)
    quantized = torch.empty_like(weight, dtype=torch.int8)
    scales = torch.empty(OUTPUT, device=weight.device, dtype=torch.float32)
    selected: list[float] = []
    for begin in range(0, OUTPUT, row_chunk):
        end = min(begin + row_chunk, OUTPUT)
        rows = weight[begin:end].float()
        base = rows.abs().amax(dim=1).clamp_min(1.0e-12) / 7.0
        candidate_scale = factors[:, None, None] * base[None, :, None]
        candidate_q = torch.round(rows[None, :, :] / candidate_scale).clamp(-7, 7)
        error = ((candidate_q * candidate_scale - rows[None, :, :]) ** 2).mean(dim=2)
        best = error.argmin(dim=0)
        current_scale = base * factors.index_select(0, best)
        quantized[begin:end] = torch.round(rows / current_scale[:, None]).clamp(-7, 7).to(
            torch.int8
        )
        scales[begin:end] = current_scale
        selected.extend(factors.index_select(0, best).cpu().tolist())
    packed = pack_signed(quantized, torch)
    if not bool(torch.equal(unpack_signed(packed, WIDTH, torch), quantized)):
        raise ExperimentError("weight packed round trip failed")
    return packed, scales.contiguous(), quantized, quantized.to(torch.int32).sum(dim=1).contiguous(), {
        "clip_factor_min": min(selected),
        "clip_factor_median": statistics.median(selected),
        "clip_factor_max": max(selected),
        "packed_bytes": int(packed.numel()),
    }


@dataclass
class QuantizedActivation:
    packed: Any
    scale: Any
    zero_point: Any


def quantize_pack_activation(tensor: Any, torch: Any) -> QuantizedActivation:
    if tensor.ndim != 2 or int(tensor.shape[1]) != WIDTH or not tensor.is_contiguous():
        raise ExperimentError("activation quantizer expects contiguous [M,12288]")
    minimum, maximum = torch.aminmax(tensor, dim=1)
    scale = ((maximum.float() - minimum.float()) / 15.0).clamp_min(1.0e-12).contiguous()
    zero_point = torch.round(-minimum.float() / scale).clamp(0, 15).contiguous()
    packed = torch.empty((int(tensor.shape[0]), WIDTH // 2), device=tensor.device,
                         dtype=torch.uint8)
    triton, _, _, pack_kernel, _ = _kernel_bundle()
    pack_kernel[(int(tensor.shape[0]), triton.cdiv(WIDTH // 2, 256))](
        tensor,
        scale,
        zero_point,
        packed,
        columns=WIDTH,
        BLOCK_PACKED=256,
        num_warps=4,
    )
    return QuantizedActivation(packed=packed, scale=scale, zero_point=zero_point)


def dequantize_corrected(accumulated: Any, activation: QuantizedActivation, weight_sum: Any,
                         weight_scale: Any, torch: Any):
    rows, columns = map(int, accumulated.shape)
    output = torch.empty((rows, columns), device=accumulated.device, dtype=torch.bfloat16)
    triton, _, _, _, kernel = _kernel_bundle()
    kernel[(rows, triton.cdiv(columns, 256))](
        accumulated,
        activation.scale,
        activation.zero_point,
        weight_sum,
        weight_scale,
        output,
        columns=columns,
        BLOCK_N=256,
        num_warps=4,
    )
    return output


def load_extension(path: Path, expected_sha256: str):
    if sha256_file(path) != expected_sha256:
        raise ExperimentError("native W4A4 extension digest differs")
    spec = importlib.util.spec_from_file_location("phase_a_w4a4_cuda", path)
    if spec is None or spec.loader is None:
        raise ExperimentError("cannot load native W4A4 extension")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "matmul", None)):
        raise ExperimentError("native extension lacks packed matmul")
    return module


class SegmentRunner:
    def __init__(self, method: str, layer: Any, extension: Any, permutation: Any | None,
                 torch: Any):
        self.method = method
        self.extension = extension
        self.torch = torch
        base_method = method
        if method.startswith("perq_"):
            if permutation is None:
                raise ExperimentError("PeRQ segment lacks its frozen permutation")
            base_method = method.removeprefix("perq_").replace("h", "local_h")
            self.gate_weight = layer.mlp.gate_proj.weight.detach().index_select(0, permutation).contiguous()
            self.up_weight = layer.mlp.up_proj.weight.detach().index_select(0, permutation).contiguous()
            down = layer.mlp.down_proj.weight.detach().index_select(1, permutation).float().contiguous()
        else:
            self.gate_weight = layer.mlp.gate_proj.weight.detach()
            self.up_weight = layer.mlp.up_proj.weight.detach()
            down = layer.mlp.down_proj.weight.detach().float().contiguous()
        self.transform = Transform(base_method, torch)
        transformed_weight = self.transform(down)
        packed, scales, quantized, weight_sum, metadata = quantize_weight(
            transformed_weight, torch
        )
        self.packed_weight = packed
        self.weight_scale = scales
        self.quantized_weight = quantized
        self.weight_sum = weight_sum
        self.weight_metadata = metadata
        del down, transformed_weight

    def producer(self, hidden: Any):
        torch = self.torch
        gate = torch.nn.functional.linear(hidden, self.gate_weight)
        up = torch.nn.functional.linear(hidden, self.up_weight)
        return (torch.nn.functional.silu(gate) * up).contiguous()

    def transform_quantize_pack(self, activation: Any):
        transformed = self.transform(activation)
        return quantize_pack_activation(transformed.contiguous(), self.torch)

    def downstream(self, activation: Any):
        quantized = self.transform_quantize_pack(activation)
        accumulated = self.extension.matmul(quantized.packed, self.packed_weight)
        return dequantize_corrected(
            accumulated, quantized, self.weight_sum, self.weight_scale, self.torch
        )

    def complete(self, hidden: Any):
        return self.downstream(self.producer(hidden))


class PackedDown:
    def __init__(self, original: Any, method: str, extension: Any, torch: Any):
        super().__init__()

    @staticmethod
    def make(original: Any, method: str, extension: Any, torch: Any):
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                if original.bias is not None or tuple(original.weight.shape) != (OUTPUT, WIDTH):
                    raise ExperimentError("Qwen down projection contract differs")
                self.in_features = WIDTH
                self.out_features = OUTPUT
                self.extension = extension
                self.transform = Transform(method, torch)
                transformed = self.transform(original.weight.detach().float().contiguous())
                packed, scales, _quantized, sums, metadata = quantize_weight(transformed, torch)
                self.register_buffer("packed_weight", packed)
                self.register_buffer("weight_scale", scales)
                self.register_buffer("weight_sum", sums)
                self.weight_metadata = metadata

            def forward(self, inputs: Any):
                original_shape = inputs.shape
                flattened = inputs.reshape(-1, WIDTH).contiguous()
                transformed = self.transform(flattened)
                quantized = quantize_pack_activation(transformed.contiguous(), torch)
                accumulated = self.extension.matmul(quantized.packed, self.packed_weight)
                output = dequantize_corrected(
                    accumulated, quantized, self.weight_sum, self.weight_scale, torch
                )
                return output.reshape(*original_shape[:-1], OUTPUT)

        return Module()


def replace_down_projections(model: Any, method: str, extension: Any, torch: Any) -> list[dict[str, object]]:
    metadata = []
    for index, layer in enumerate(model.model.layers):
        replacement = PackedDown.make(layer.mlp.down_proj, method, extension, torch)
        layer.mlp.down_proj = replacement
        metadata.append({"layer": index, **replacement.weight_metadata})
    if len(metadata) != 36:
        raise ExperimentError("expected 36 replaced down projections")
    return metadata


def cuda_sample(function: Callable[[], Any], torch: Any) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = function()
    end.record()
    end.synchronize()
    value = float(start.elapsed_time(end))
    del result
    return value


def warmup(function: Callable[[], Any], count: int, torch: Any) -> None:
    for _ in range(count):
        result = function()
        del result
    torch.cuda.synchronize()


def interleaved_measure(functions: Mapping[str, Callable[[], Any]], samples: int,
                        warmups: int, torch: Any) -> tuple[dict[str, object], list[dict[str, object]]]:
    for function in functions.values():
        warmup(function, warmups, torch)
    raw = {name: [] for name in functions}
    names = list(functions)
    rows = []
    for repetition in range(samples):
        order = names if repetition % 2 == 0 else list(reversed(names))
        for order_index, name in enumerate(order):
            value = cuda_sample(functions[name], torch)
            raw[name].append(value)
            rows.append({
                "method": name,
                "repetition": repetition,
                "order_index": order_index,
                "milliseconds": value,
            })
    return {name: summarize(values) for name, values in raw.items()}, rows


def validate_native_runner(runner: SegmentRunner, hidden: Any, torch: Any) -> dict[str, object]:
    activation = runner.producer(hidden[: min(8, int(hidden.shape[0]))])
    transformed = runner.transform(activation)
    state = quantize_pack_activation(transformed.contiguous(), torch)
    minimum, maximum = torch.aminmax(transformed, dim=1)
    scale = ((maximum.float() - minimum.float()) / 15.0).clamp_min(1.0e-12)
    zp = torch.round(-minimum.float() / scale).clamp(0, 15)
    quantized = torch.round(transformed.float() / scale[:, None] + zp[:, None]).clamp(0, 15)
    unpacked = unpack_signed(state.packed, WIDTH, torch).to(torch.int16) + 8
    if not bool(torch.equal(unpacked, quantized.to(torch.int16))):
        raise ExperimentError(f"{runner.method} asymmetric packed round trip failed")
    accumulated = runner.extension.matmul(state.packed, runner.packed_weight)
    corrected = accumulated + (
        (8 - state.zero_point.to(torch.int32))[:, None] * runner.weight_sum[None, :]
    )
    reference = (quantized - zp[:, None]) @ runner.quantized_weight.float().T
    integer_error = float((corrected.float() - reference).abs().max().item())
    if integer_error != 0.0:
        raise ExperimentError(f"{runner.method} native corrected integer GEMM differs")
    output = dequantize_corrected(
        accumulated, state, runner.weight_sum, runner.weight_scale, torch
    )
    expected = (reference * scale[:, None] * runner.weight_scale[None, :]).to(torch.bfloat16)
    dequant_error = float((output.float() - expected.float()).abs().max().item())
    dequant_tolerance = max(1.0e-3, float(expected.float().abs().max().item()) / 64.0)
    if dequant_error > dequant_tolerance:
        raise ExperimentError(f"{runner.method} corrected dequantization differs")
    return {
        "rows": int(activation.shape[0]),
        "activation_packed_dtype": str(state.packed.dtype),
        "activation_packed_shape": list(state.packed.shape),
        "activation_packed_bytes": int(state.packed.numel()),
        "weight_packed_dtype": str(runner.packed_weight.dtype),
        "weight_packed_shape": list(runner.packed_weight.shape),
        "weight_packed_bytes": int(runner.packed_weight.numel()),
        "integer_max_abs_error": integer_error,
        "dequant_max_abs_error": dequant_error,
        "dequant_tolerance": dequant_tolerance,
    }


def read_prompt(contract: Mapping[str, Any], torch: Any, transformers: Any, pyarrow: Any):
    arrow = Path(contract["inputs"]["prompt_source"]).resolve(strict=True)
    if sha256_file(arrow) != contract["inputs"]["prompt_source_sha256"]:
        raise ExperimentError("prompt Arrow digest differs")
    with pyarrow.memory_map(str(arrow), "r") as source:
        table = pyarrow.ipc.open_stream(source).read_all()
    texts = table.column("text").to_pylist()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        contract["model"]["snapshot"], local_files_only=True, trust_remote_code=False,
        use_fast=True,
    )
    encoded = tokenizer("\n\n".join(texts), add_special_tokens=False, return_tensors="pt").input_ids
    if int(encoded.shape[1]) < 2048:
        raise ExperimentError("validation prompt is shorter than 2K")
    prompt = encoded[:, :2048].contiguous()
    return prompt, {
        "tokens": 2048,
        "sha256": sha256_bytes(prompt.numpy().tobytes()),
        "source": str(arrow),
        "source_sha256": sha256_file(arrow),
        "add_special_tokens": False,
    }


def load_model(contract: Mapping[str, Any], transformers: Any, torch: Any):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        contract["model"]["snapshot"],
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).eval().to("cuda:0")
    observed = (
        type(model).__name__,
        int(model.config.num_hidden_layers),
        int(model.config.hidden_size),
        int(model.config.intermediate_size),
    )
    if observed != ("Qwen3ForCausalLM", 36, HIDDEN, WIDTH):
        raise ExperimentError(f"loaded model contract differs: {observed}")
    return model


def capture_hidden_inputs(model: Any, prompt: Any, torch: Any) -> dict[str, Any]:
    captured: list[Any] = []

    def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
        captured.append(inputs[0].detach())

    handle = model.model.layers[0].mlp.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            prefill = model.model(input_ids=prompt.to("cuda:0"), use_cache=True, return_dict=True)
        torch.cuda.synchronize()
        if len(captured) != 1 or tuple(captured[0].shape) != (1, 2048, HIDDEN):
            raise ExperimentError("prefill target-layer producer input shape differs")
        prefill_hidden = captured.pop().reshape(2048, HIDDEN).contiguous()
        next_token = prompt[:, -1:].to("cuda:0")
        with torch.inference_mode():
            decoded = model.model(
                input_ids=next_token,
                past_key_values=prefill.past_key_values,
                use_cache=True,
                return_dict=True,
            )
        del decoded
        torch.cuda.synchronize()
        if len(captured) != 1 or tuple(captured[0].shape) != (1, 1, HIDDEN):
            raise ExperimentError("decode target-layer producer input shape differs")
        decode_one = captured.pop().reshape(1, HIDDEN).contiguous()
    finally:
        handle.remove()
    return {
        "prefill_b1_s2048": prefill_hidden,
        "decode_b1_ctx2048": decode_one,
        "decode_b8_ctx2048": decode_one.repeat(8, 1).contiguous(),
    }


def benchmark_segments(model: Any, hidden: Mapping[str, Any], cache_path: Path, extension: Any,
                       contract: Mapping[str, Any], torch: Any):
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    if cache.get("schema_version") != "structured-rotations-v2-gate-a-activation-sample-v1":
        raise ExperimentError("A1 activation cache schema differs")
    permutations = {
        int(block): torch.tensor(values, device="cuda:0", dtype=torch.long)
        for block, values in cache["massdiff_permutations"].items()
    }
    layer = model.model.layers[0]
    runners = {}
    validations = {}
    for method in SEGMENT_METHODS:
        permutation = None
        if method == "perq_h32":
            permutation = permutations[32]
        elif method == "perq_h128":
            permutation = permutations[128]
        runners[method] = SegmentRunner(method, layer, extension, permutation, torch)
        validations[method] = validate_native_runner(
            runners[method], hidden["decode_b8_ctx2048"], torch
        )
    measurements: dict[str, object] = {}
    raw_rows = []
    for workload, rows in hidden.items():
        samples = (
            contract["timing"]["segment_samples_prefill"]
            if workload == "prefill_b1_s2048"
            else contract["timing"]["segment_samples_decode"]
        )
        produced = {name: runner.producer(rows) for name, runner in runners.items()}
        metric_functions = {
            "complete_segment": {name: (lambda runner=runner: runner.complete(rows))
                                 for name, runner in runners.items()},
            "transform_quantize_pack_consumer": {
                name: (lambda runner=runner, activation=produced[name]: runner.downstream(activation))
                for name, runner in runners.items()
            },
            "transform_quantize_pack": {
                name: (lambda runner=runner, activation=produced[name]:
                       runner.transform_quantize_pack(activation))
                for name, runner in runners.items()
            },
            "transform_only": {
                name: (lambda runner=runner, activation=produced[name]:
                       runner.transform(activation))
                for name, runner in runners.items()
            },
        }
        measurements[workload] = {}
        for metric, functions in metric_functions.items():
            summaries, raw = interleaved_measure(
                functions, samples, contract["timing"]["segment_warmups"], torch
            )
            measurements[workload][metric] = summaries
            raw_rows.extend({"execution_kind": "native_segment", "workload": workload,
                             "metric": metric, **row} for row in raw)
        reference_summary, reference_raw = interleaved_measure(
            {"torch_full_reference": lambda activation=produced["full_hadamard"]:
             torch_full_reference(activation, torch)},
            samples,
            contract["timing"]["segment_warmups"],
            torch,
        )
        measurements[workload]["original_transform_control"] = reference_summary
        raw_rows.extend({"execution_kind": "supporting_transform_only", "workload": workload,
                         "metric": "original_transform_control", **row}
                        for row in reference_raw)
    return measurements, validations, raw_rows


def prefill_call(model: Any, prompt: Any):
    return model(input_ids=prompt, use_cache=True, return_dict=True, logits_to_keep=1)


def decode_group(model: Any, prompt: Any, batch: int, tokens: int, torch: Any):
    with torch.inference_mode():
        initial = prefill_call(model, prompt)
        cache = initial.past_key_values
        next_token = initial.logits[:, -1:].argmax(dim=-1)
        if batch != 1:
            cache.batch_repeat_interleave(batch)
            next_token = next_token.repeat_interleave(batch, dim=0)
        for _ in range(tokens):
            output = model(
                input_ids=next_token,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
            cache = output.past_key_values
            next_token = output.logits[:, -1:].argmax(dim=-1)
    return next_token


def cuda_decode_sample(model: Any, prompt: Any, batch: int, tokens: int, torch: Any) -> float:
    # Cache construction is deliberately outside the decode timing boundary.
    with torch.inference_mode():
        initial = prefill_call(model, prompt)
        cache = initial.past_key_values
        next_token = initial.logits[:, -1:].argmax(dim=-1)
        if batch != 1:
            cache.batch_repeat_interleave(batch)
            next_token = next_token.repeat_interleave(batch, dim=0)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(tokens):
            output = model(
                input_ids=next_token,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
            cache = output.past_key_values
            next_token = output.logits[:, -1:].argmax(dim=-1)
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end)) / tokens


def benchmark_e2e_visit(model: Any, prompt: Any, contract: Mapping[str, Any], torch: Any):
    samples = int(contract["timing"]["end_to_end_samples_per_visit"])
    with torch.inference_mode():
        warmup(lambda: prefill_call(model, prompt), 1, torch)
        prefill = [cuda_sample(lambda: prefill_call(model, prompt), torch) for _ in range(samples)]
        values = {"prefill_b1_s2048": prefill}
        for workload, batch in (("decode_b1_ctx2048", 1), ("decode_b8_ctx2048", 8)):
            tokens = int(contract["workloads"][workload]["timed_tokens"])
            warmup(lambda batch=batch, tokens=tokens: decode_group(
                model, prompt, batch, tokens, torch
            ), 1, torch)
            values[workload] = [
                cuda_decode_sample(model, prompt, batch, tokens, torch) for _ in range(samples)
            ]
    return values


def benchmark_end_to_end(contract: Mapping[str, Any], prompt: Any, extension: Any,
                         transformers: Any, torch: Any, events: Path):
    raw: dict[str, dict[str, list[float]]] = {
        workload: {method: [] for method in E2E_METHODS}
        for workload in contract["workloads"]
    }
    preparation = []
    visits = int(contract["timing"]["end_to_end_visits"])
    for visit in range(visits):
        methods = E2E_METHODS if visit % 2 == 0 else tuple(reversed(E2E_METHODS))
        for order_index, method in enumerate(methods):
            append_event(events, "end_to_end_visit_started", visit=visit, method=method,
                         order_index=order_index)
            started = time.monotonic()
            model = load_model(contract, transformers, torch)
            weight_metadata = replace_down_projections(model, method, extension, torch)
            torch.cuda.synchronize()
            preparation.append({
                "visit": visit,
                "method": method,
                "seconds": time.monotonic() - started,
                "layers": len(weight_metadata),
                "clip_factor_min": min(row["clip_factor_min"] for row in weight_metadata),
                "clip_factor_max": max(row["clip_factor_max"] for row in weight_metadata),
            })
            values = benchmark_e2e_visit(model, prompt, contract, torch)
            for workload, samples in values.items():
                raw[workload][method].extend(samples)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            append_event(events, "end_to_end_visit_completed", visit=visit, method=method)
    summaries = {
        workload: {method: summarize(samples) for method, samples in methods.items()}
        for workload, methods in raw.items()
    }
    raw_rows = [
        {
            "execution_kind": "native_end_to_end",
            "workload": workload,
            "method": method,
            "repetition": repetition,
            "milliseconds": value,
            "unit": "ms_per_prefill" if workload == "prefill_b1_s2048" else "ms_per_token",
        }
        for workload, methods in raw.items()
        for method, values in methods.items()
        for repetition, value in enumerate(values)
    ]
    return summaries, preparation, raw_rows


def environment_record(torch: Any) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(0)
    driver = run_text([
        "nvidia-smi",
        "--query-gpu=driver_version,name,uuid,memory.total",
        "--format=csv,noheader,nounits",
    ])
    return {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": __import__("triton").__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_name": properties.name,
        "device_uuid": str(getattr(properties, "uuid", "unavailable")),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "total_memory_bytes": int(properties.total_memory),
        "driver_query": driver,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
    }


def verify_model_hashes(quality: Mapping[str, Any]) -> dict[str, object]:
    snapshot = Path(quality["model"]["snapshot"]).resolve(strict=True)
    files = {
        "config.json": quality["model"]["config_sha256"],
        "model.safetensors.index.json": quality["model"]["index_sha256"],
        "tokenizer.json": quality["model"]["tokenizer_sha256"],
        "tokenizer_config.json": quality["model"]["tokenizer_config_sha256"],
    }
    for index, digest in enumerate(quality["model"]["weight_shards_sha256"], start=1):
        files[f"model-{index:05d}-of-00005.safetensors"] = digest
    observed = {}
    for name, expected in files.items():
        actual = sha256_file(snapshot / name)
        if actual != expected:
            raise ExperimentError(f"model input digest differs for {name}")
        observed[name] = actual
    return observed


def make_ledger(result: Mapping[str, Any]) -> list[dict[str, object]]:
    common = {
        "run_id": result["run_id"],
        "code_commit": result["stage"]["head"],
        "model_revision": result["model"]["revision"],
        "target_sites": "all 36 FFN down_proj inputs for end-to-end; layer 0 for segment",
        "permutation_seed": 0,
        "calibration_hash": result["inputs"]["quality_layer0_cache_sha256"],
        "weight_quantizer": "symmetric narrow INT4; per-output-channel 21-factor MSE clipping",
        "activation_quantizer": "dynamic per-token asymmetric unsigned INT4",
        "scale_groups": "weight per output channel; activation per token row",
        "clipping_budget": "weight factors 1.00..0.80 inclusive; activation observed min/max",
        "gpu_model": result["environment"]["device_name"],
        "gpu_count": 1,
        "mean_output_error": "not_run",
        "tail_output_error": "not_run",
        "dev_nll": "not_run",
        "final_ppl_if_run": "not_run",
        "metadata_bytes": "not_run",
        "active_gpu_hours": result["active_gpu_hours"],
        "allocated_gpu_hours": "pending_terminal_accounting",
        "status": "complete",
        "next_gate": "Decision A",
    }
    rows = []
    for method in SEGMENT_METHODS:
        for workload, metrics in result["segment"].items():
            rows.append({
                **common,
                "method": method,
                "transform_definition": method,
                "b": 32 if method.endswith("32") else 128 if method.endswith("128") else (
                    WIDTH if method == "full_hadamard" else "not_applicable"
                ),
                "r": "not_run",
                "execution_kind": "native_segment",
                "workload": workload,
                "segment_ms": metrics["complete_segment"][method]["median_ms"],
                "end_to_end_ms": "not_run",
                "timing_dispersion": metrics["complete_segment"][method][
                    "median_absolute_deviation_ms"
                ],
            })
    for method in E2E_METHODS:
        for workload, metrics in result["end_to_end"].items():
            rows.append({
                **common,
                "method": method,
                "transform_definition": method,
                "b": 32 if method.endswith("32") else 128 if method.endswith("128") else (
                    WIDTH if method == "full_hadamard" else "not_applicable"
                ),
                "r": "not_run",
                "execution_kind": "native_end_to_end",
                "workload": workload,
                "segment_ms": "not_run",
                "end_to_end_ms": metrics[method]["median_ms"],
                "timing_dispersion": metrics[method]["median_absolute_deviation_ms"],
            })
    return rows


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    root = Path(run_text(["git", "rev-parse", "--show-toplevel"])).resolve(strict=True)
    contract_path = args.contract.resolve(strict=True)
    quality_path = root / "experiments/structured_rotations_v2/gate_a/contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    validate_contract(contract, quality)
    stage = verify_stage(root, contract_path, quality_path)
    result_root = Path(contract["storage"]["retained_root"]).resolve()
    result_directory = args.result_directory.resolve()
    if result_directory.parent != result_root / "runs":
        raise ExperimentError("result directory is outside the frozen retained run root")
    result_directory.mkdir(mode=0o700)
    events = result_directory / "events.jsonl"
    events.touch(mode=0o600, exist_ok=False)
    append_event(events, "run_started", argv=sys.argv, stage=stage)

    import pyarrow
    import torch
    import transformers

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ExperimentError("A2 requires exactly one visible CUDA device")
    environment = environment_record(torch)
    if environment["compute_capability"] != [8, 6] or "RTX 3090" not in environment["device_name"]:
        raise ExperimentError(f"native artifact target differs: {environment}")
    append_event(events, "environment_verified", environment=environment)
    model_hashes = verify_model_hashes(quality)
    extension_path = root / contract["backend"]["artifact"]
    extension = load_extension(extension_path, contract["backend"]["artifact_sha256"])
    cache_path = root / contract["inputs"]["quality_layer0_cache"]
    if sha256_file(cache_path) != contract["inputs"]["quality_layer0_cache_sha256"]:
        raise ExperimentError("frozen A1 layer-0 cache digest differs")
    prompt, prompt_record = read_prompt(contract, torch, transformers, pyarrow)
    append_event(events, "inputs_verified", prompt=prompt_record)

    model = load_model(contract, transformers, torch)
    hidden = capture_hidden_inputs(model, prompt, torch)
    shapes = {name: list(value.shape) for name, value in hidden.items()}
    expected_shapes = {
        "prefill_b1_s2048": [2048, HIDDEN],
        "decode_b1_ctx2048": [1, HIDDEN],
        "decode_b8_ctx2048": [8, HIDDEN],
    }
    if shapes != expected_shapes:
        raise ExperimentError(f"actual producer-input shapes differ: {shapes}")
    append_event(events, "tensor_contract_verified", shapes=shapes)
    segment, native_validation, segment_raw = benchmark_segments(
        model, hidden, cache_path, extension, contract, torch
    )
    del model, hidden
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    append_event(events, "native_segment_completed")

    end_to_end, preparation, e2e_raw = benchmark_end_to_end(
        contract, prompt.to("cuda:0"), extension, transformers, torch, events
    )
    active_gpu_hours = (time.monotonic() - started) / 3600.0
    result = {
        "schema_version": SCHEMA,
        "run_id": f"slurm-{os.environ['SLURM_JOB_ID']}",
        "recorded_at_utc": utc_now(),
        "stage": stage,
        "execution_contract_sha256": sha256_file(contract_path),
        "quality_contract_source_sha256": contract["quality_contract"]["sha256"],
        "quality_result_sha256": contract["quality_contract"]["result_sha256"],
        "model": {
            "repository": contract["model"]["repository"],
            "revision": contract["model"]["revision"],
            "dtype": contract["model"]["dtype"],
            "attention": contract["model"]["attention"],
        },
        "environment": environment,
        "inputs": {
            "model_hashes": model_hashes,
            "prompt": prompt_record,
            "quality_layer0_cache_sha256": sha256_file(cache_path),
            "native_extension_sha256": sha256_file(extension_path),
        },
        "tensor_contract": contract["tensor_contract"],
        "actual_producer_input_shapes": shapes,
        "native_validation": native_validation,
        "segment": segment,
        "end_to_end": end_to_end,
        "end_to_end_preparation": preparation,
        "execution_kind": ["native_segment", "native_end_to_end"],
        "fusion": contract["backend"]["fusion"],
        "perq_limitation": contract["methods"]["perq_limitation"],
        "active_gpu_hours": active_gpu_hours,
        "allocated_gpu_hours": "pending_terminal_accounting",
        "confounds": [
            "A1 supplies layer-output quality, while A2 supplies native timing; A2 does not rerun quality.",
            "PeRQ is segment-only because only six A1 layer permutations were frozen.",
            "Decode batch 8 repeats one real 2K cache across identical requests before timing.",
            "Transform, quantize-pack, consumer, and correction-dequant are equally separate for all methods.",
        ],
        "next_experiment": contract["next_experiment"],
    }
    write_json(result_directory / "results.json", result)
    write_jsonl(result_directory / "raw-timings.jsonl", [*segment_raw, *e2e_raw])
    write_jsonl(result_directory / "ledger.jsonl", make_ledger(result))
    append_event(events, "run_completed", active_gpu_hours=active_gpu_hours)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--contract", type=Path, required=True)
    value.add_argument("--result-directory", type=Path, required=True)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = run(args)
    print(json.dumps({
        "run_id": result["run_id"],
        "result_directory": str(args.result_directory),
        "active_gpu_hours": result["active_gpu_hours"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
