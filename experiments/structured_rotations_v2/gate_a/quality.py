#!/usr/bin/env python3
"""One-shot Structured Rotations v2 Gate A quality experiment.

This program intentionally produces numerical-only W4A4 reconstruction evidence.
It does not time a transform, invoke a native low-bit consumer, screen bridges,
or select transforms per layer.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "structured-rotations-v2-gate-a-quality-v1"
EXPECTED_CONTRACT_SCHEMA = "structured-rotations-v2-gate-a-contract-v1"
METHOD_NAMES = (
    "identity",
    "full_hadamard",
    "local_h32",
    "local_h128",
    "perq_h32",
    "perq_h128",
)


class ExperimentError(RuntimeError):
    """The frozen experiment contract was violated or the run is incomplete."""


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
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
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


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != EXPECTED_CONTRACT_SCHEMA:
        raise ExperimentError("contract schema differs")
    if contract["scope"] != {
        "experiment": "A1",
        "target": "FFN down-projection inputs only",
        "execution_kind": "numerical_only",
        "excluded": [
            "native timing",
            "Gate B representative bridges",
            "transform-family search",
            "per-layer transform selection",
            "whole-model quality evaluation",
        ],
    }:
        raise ExperimentError("A1 scope or exclusions differ")
    model = contract["model"]
    if model["substitution"] != "Qwen/Qwen3-8B":
        raise ExperimentError("the frozen permitted model substitution differs")
    if model["revision"] != "b968826d9c46dd6066d109eabc6255188de91218":
        raise ExperimentError("model revision differs")
    if model["sampled_layers"] != [0, 7, 14, 21, 28, 35]:
        raise ExperimentError("six-layer sample differs")
    if model["ffn_intermediate_size"] != 12288:
        raise ExperimentError("FFN width differs")
    data = contract["data"]
    expected_data = {
        "calibration_tokens": 32768,
        "development_tokens": 16384,
        "sequence_length": 2048,
        "calibration_split": "train",
        "development_split": "validation",
    }
    if any(data[key] != value for key, value in expected_data.items()):
        raise ExperimentError("calibration/development split contract differs")
    if tuple(method["name"] for method in contract["methods"]) != METHOD_NAMES:
        raise ExperimentError("six fixed controls differ")
    quantization = contract["quantization"]
    if quantization["group_width"] != 12288:
        raise ExperimentError("quantizer group width differs")
    if quantization["rounding"] != "round to nearest, ties to even":
        raise ExperimentError("rounding mode differs")
    if contract["budget"]["attempt_limit"] != 1 or contract["budget"]["requeue"]:
        raise ExperimentError("one-shot/no-requeue contract differs")


def verify_stage(contract_path: Path, contract: Mapping[str, Any]) -> dict[str, object]:
    root = Path(run_text(["git", "rev-parse", "--show-toplevel"], cwd=contract_path.parent)).resolve()
    expected_parent = Path(contract["storage"]["staging_parent"]).resolve()
    if root.parent != expected_parent:
        raise ExperimentError(f"run root {root} is not a direct timestamped stage under {expected_parent}")
    if not (root / ".git").is_dir():
        raise ExperimentError("stage does not contain ordinary self-contained .git metadata")
    metadata_path = root / "REPRODUCIBILITY_METADATA.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if Path(metadata["staged_repo"]).resolve() != root:
        raise ExperimentError("stage metadata does not bind the current repository")
    if Path(metadata["source_repo"]).resolve() == root:
        raise ExperimentError("stage metadata source and target are identical")
    alternates = root / ".git" / "objects" / "info" / "alternates"
    if alternates.exists():
        raise ExperimentError("stage Git objects depend on alternates")
    run_text(["git", "fsck", "--connectivity-only", "--no-progress"], cwd=root)
    status = run_text(["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=root)
    status_lines = [line for line in status.splitlines() if line]
    if status_lines != ["?? REPRODUCIBILITY_METADATA.json"]:
        raise ExperimentError(f"stage has unexpected working-tree state: {status_lines!r}")
    return {
        "root": str(root),
        "head": run_text(["git", "rev-parse", "HEAD"], cwd=root),
        "tree": run_text(["git", "rev-parse", "HEAD^{tree}"], cwd=root),
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "contract_path": str(contract_path.resolve()),
        "contract_sha256": sha256_file(contract_path),
        "git_status": status_lines,
    }


def _paley_hadamard_12_rows() -> list[list[int]]:
    prime = 11
    squares = {value * value % prime for value in range(1, prime)}

    def character(value: int) -> int:
        value %= prime
        if value == 0:
            return 0
        return 1 if value in squares else -1

    matrix = [[1] * 12]
    for row in range(prime):
        matrix.append([-1] + [1 if row == column else character(row - column)
                              for column in range(prime)])
    return matrix


def paley_hadamard_12(torch: Any, *, device: object = "cpu", dtype: object | None = None):
    dtype = dtype or torch.float64
    return torch.tensor(_paley_hadamard_12_rows(), device=device, dtype=dtype) / math.sqrt(12)


def _normalized_fht_last(tensor: Any, torch: Any):
    width = int(tensor.shape[-1])
    if width <= 0 or width & (width - 1):
        raise ExperimentError("Sylvester width must be a positive power of two")
    work = tensor.clone()
    half = 1
    while half < width:
        grouped = work.reshape(*work.shape[:-1], -1, 2 * half)
        lhs = grouped[..., :half].clone()
        rhs = grouped[..., half:].clone()
        grouped[..., :half] = lhs + rhs
        grouped[..., half:] = lhs - rhs
        half *= 2
    return work / math.sqrt(width)


def block_hadamard(tensor: Any, block_size: int, torch: Any):
    width = int(tensor.shape[-1])
    if width % block_size:
        raise ExperimentError(f"width {width} is not divisible by block size {block_size}")
    original_shape = tuple(tensor.shape)
    blocks = tensor.reshape(*original_shape[:-1], width // block_size, block_size)
    return _normalized_fht_last(blocks, torch).reshape(original_shape)


def full_hadamard_12288(tensor: Any, torch: Any):
    if int(tensor.shape[-1]) != 12288:
        raise ExperimentError("the frozen exact full transform requires width 12288")
    original_shape = tuple(tensor.shape)
    work = tensor.reshape(*original_shape[:-1], 12, 1024)
    work = _normalized_fht_last(work, torch)
    outer = paley_hadamard_12(torch, device=tensor.device, dtype=tensor.dtype)
    work = torch.einsum("...ak,ab->...bk", work, outer)
    return work.reshape(original_shape)


def transform_tensor(tensor: Any, method: str, permutation: Any | None, torch: Any):
    if method == "identity":
        return tensor
    if method == "full_hadamard":
        return full_hadamard_12288(tensor, torch)
    if method in ("local_h32", "local_h128", "perq_h32", "perq_h128"):
        block_size = 32 if method.endswith("32") else 128
        work = tensor.index_select(-1, permutation) if method.startswith("perq") else tensor
        return block_hadamard(work, block_size, torch)
    raise ExperimentError(f"unknown method {method}")


def massdiff_permutation(mean_abs: Any, channel_p99: Any, block_size: int) -> list[int]:
    """Greedy PeRQ MassDiff membership plus frozen p99 ordering within blocks."""

    import heapq

    if mean_abs.ndim != 1 or channel_p99.ndim != 1 or mean_abs.shape != channel_p99.shape:
        raise ExperimentError("MassDiff statistics must be equal one-dimensional vectors")
    width = int(mean_abs.numel())
    if width % block_size:
        raise ExperimentError("MassDiff width is not divisible by block size")
    means = [float(value) for value in mean_abs.tolist()]
    tails = [float(value) for value in channel_p99.tolist()]
    if any(not math.isfinite(value) or value < 0 for value in means + tails):
        raise ExperimentError("MassDiff statistics must be finite and nonnegative")
    ordered_channels = sorted(range(width), key=lambda index: (-means[index], index))
    blocks: list[list[int]] = [[] for _ in range(width // block_size)]
    heap = [(0.0, block) for block in range(len(blocks))]
    heapq.heapify(heap)
    for channel in ordered_channels:
        mass, block = heapq.heappop(heap)
        blocks[block].append(channel)
        mass += means[channel]
        if len(blocks[block]) < block_size:
            heapq.heappush(heap, (mass, block))
    if heap or any(len(block) != block_size for block in blocks):
        raise ExperimentError("MassDiff did not fill every block exactly")
    for block in blocks:
        block.sort(key=lambda index: (-tails[index], index))
    permutation = [channel for block in blocks for channel in block]
    if sorted(permutation) != list(range(width)):
        raise ExperimentError("MassDiff result is not a permutation")
    return permutation


def pack_int4(values: Any, *, signed: bool, torch: Any):
    if values.shape[-1] % 2:
        raise ExperimentError("INT4 packing requires an even final dimension")
    if signed:
        if bool(((values < -8) | (values > 7)).any().item()):
            raise ExperimentError("signed INT4 value is out of range")
        encoded = values.to(torch.int16) & 0xF
    else:
        if bool(((values < 0) | (values > 15)).any().item()):
            raise ExperimentError("unsigned INT4 value is out of range")
        encoded = values.to(torch.int16)
    return (encoded[..., 0::2] | (encoded[..., 1::2] << 4)).to(torch.uint8)


def unpack_int4(packed: Any, *, columns: int, signed: bool, torch: Any):
    if columns <= 0 or columns % 2 or int(packed.shape[-1]) * 2 != columns:
        raise ExperimentError("packed INT4 shape differs")
    output = torch.empty(*packed.shape[:-1], columns, device=packed.device, dtype=torch.int8)
    output[..., 0::2] = (packed & 0xF).to(torch.int8)
    output[..., 1::2] = (packed >> 4).to(torch.int8)
    if signed:
        output = torch.where(output >= 8, output - 16, output)
    return output


def quantize_weight_reconstruction(weight: Any, torch: Any, *, row_chunk: int = 16):
    """Matched per-output-channel symmetric INT4 with a bounded MSE clip search."""

    factors = torch.linspace(1.0, 0.8, 21, device=weight.device, dtype=torch.float32)
    reconstructed = torch.empty_like(weight, dtype=torch.float32)
    chosen_factors: list[float] = []
    squared_error = 0.0
    for begin in range(0, int(weight.shape[0]), row_chunk):
        end = min(begin + row_chunk, int(weight.shape[0]))
        rows = weight[begin:end].float()
        base = rows.abs().amax(dim=1).clamp_min(1.0e-12) / 7.0
        candidate_scales = factors[:, None, None] * base[None, :, None]
        candidate_q = torch.round(rows[None, :, :] / candidate_scales).clamp(-7, 7)
        candidate_error = ((candidate_q * candidate_scales - rows[None, :, :]) ** 2).mean(dim=2)
        best = candidate_error.argmin(dim=0)
        scales = base * factors.index_select(0, best)
        quantized = torch.round(rows / scales[:, None]).clamp(-7, 7).to(torch.int8)
        packed = pack_int4(quantized, signed=True, torch=torch)
        unpacked = unpack_int4(packed, columns=int(rows.shape[1]), signed=True, torch=torch)
        if not bool(torch.equal(quantized, unpacked)):
            raise ExperimentError("signed weight INT4 pack/unpack round trip failed")
        current = unpacked.float() * scales[:, None]
        reconstructed[begin:end] = current
        squared_error += float(((current - rows) ** 2).sum().item())
        chosen_factors.extend(float(value) for value in factors.index_select(0, best).cpu().tolist())
    return reconstructed, {
        "rounding": "nearest_ties_to_even",
        "alphabet": "signed_narrow_int4[-7,7]",
        "scale_granularity": "per_output_channel",
        "clip_search_factors": 21,
        "clip_factor_min": min(chosen_factors),
        "clip_factor_median": statistics.median(chosen_factors),
        "clip_factor_max": max(chosen_factors),
        "reconstruction_mse": squared_error / int(weight.numel()),
        "packed_roundtrip_verified": True,
    }


def quantize_activation_reconstruction(activation: Any, torch: Any):
    """Dynamic per-token asymmetric INT4, including a packed round trip."""

    minimum = activation.amin(dim=1)
    maximum = activation.amax(dim=1)
    scale = ((maximum - minimum) / 15.0).clamp_min(1.0e-12)
    zero_point = torch.round(-minimum / scale).clamp(0, 15)
    quantized = torch.round(activation / scale[:, None] + zero_point[:, None]).clamp(0, 15).to(
        torch.int8
    )
    packed = pack_int4(quantized, signed=False, torch=torch)
    unpacked = unpack_int4(packed, columns=int(activation.shape[1]), signed=False, torch=torch)
    if not bool(torch.equal(quantized, unpacked)):
        raise ExperimentError("activation INT4 pack/unpack round trip failed")
    reconstructed = (unpacked.float() - zero_point[:, None]) * scale[:, None]
    return reconstructed, {
        "rounding": "nearest_ties_to_even",
        "alphabet": "unsigned_int4[0,15]",
        "scale_granularity": "dynamic_per_token_row",
        "packed_roundtrip_verified": True,
        "activation_mse": float(((reconstructed - activation.float()) ** 2).mean().item()),
    }


def output_error_metrics(predicted: Any, reference: Any, torch: Any) -> dict[str, object]:
    if predicted.shape != reference.shape or predicted.ndim != 2:
        raise ExperimentError("predicted/reference output shapes differ")
    difference = predicted.float() - reference.float()
    row_numerator = difference.square().sum(dim=1)
    row_denominator = reference.float().square().sum(dim=1).clamp_min(1.0e-12)
    row_error = row_numerator / row_denominator
    tail_count = max(1, math.ceil(int(row_error.numel()) * 0.10))
    tail = torch.topk(row_error, tail_count, largest=True, sorted=False).values
    numerator = float(row_numerator.sum().item())
    denominator = float(row_denominator.sum().item())
    return {
        "rows": int(reference.shape[0]),
        "outputs": int(reference.shape[1]),
        "squared_error_sum": numerator,
        "reference_squared_sum": denominator,
        "mean_output_error": numerator / denominator,
        "mean_row_output_error": float(row_error.mean().item()),
        "tail_fraction": 0.10,
        "tail_rows": tail_count,
        "tail_output_error": float(tail.mean().item()),
    }


def representative_coverage(sample: Any, representative_channels: Sequence[int], torch: Any):
    width = int(sample.shape[1])
    mask = torch.zeros(width, dtype=torch.bool, device=sample.device)
    mask[torch.tensor(list(representative_channels), device=sample.device)] = True
    absolute = sample.float().abs()
    event_count = max(1, math.ceil(int(absolute.numel()) * 0.001))
    event_indexes = torch.topk(absolute.reshape(-1), event_count, largest=True, sorted=False).indices
    event_channels = event_indexes % width
    row_channels = absolute.argmax(dim=1)
    event_coverage = float(mask.index_select(0, event_channels).float().mean().item())
    row_coverage = float(mask.index_select(0, row_channels).float().mean().item())
    chance = len(representative_channels) / width
    return {
        "representatives": len(representative_channels),
        "representative_fraction": chance,
        "large_event_fraction": 0.001,
        "large_events": event_count,
        "large_event_coverage": event_coverage,
        "large_event_enrichment_vs_fraction": event_coverage / chance,
        "row_maximum_coverage": row_coverage,
        "row_maximum_enrichment_vs_fraction": row_coverage / chance,
    }


def _read_wikitext_arrow(path: Path, pyarrow: Any) -> list[str]:
    with pyarrow.memory_map(str(path.resolve(strict=True)), "r") as source:
        table = pyarrow.ipc.open_stream(source).read_all()
    if table.column_names != ["text"]:
        raise ExperimentError(f"WikiText Arrow columns differ: {table.column_names!r}")
    values = table.column("text").to_pylist()
    if not values or any(not isinstance(value, str) for value in values):
        raise ExperimentError("WikiText text column is malformed")
    return values


def _article_hashes(texts: Sequence[str]) -> set[str]:
    articles: list[list[str]] = []
    current: list[str] = []
    for text in texts:
        stripped = text.strip()
        # WikiText encodes article titles as ``= Title =`` and nested section
        # headings as ``= = Section = =``.  Only the former starts a document.
        is_article_title = (
            stripped.startswith("= ")
            and not stripped.startswith("= =")
            and stripped.endswith(" =")
        )
        if is_article_title and current:
            articles.append(current)
            current = []
        current.append(text)
    if current:
        articles.append(current)
    return {sha256_bytes("\n".join(article).encode()) for article in articles if any(article)}


def tokenize_stream(snapshot: Path, arrow: Path, tokens: int, *, transformers: Any, pyarrow: Any,
                    torch: Any):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True
    )
    texts = _read_wikitext_arrow(arrow, pyarrow)
    encoded = tokenizer("\n\n".join(texts), add_special_tokens=False, return_tensors="pt")
    input_ids = encoded.input_ids.cpu()
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < tokens:
        raise ExperimentError(f"token stream is shorter than {tokens}")
    selected = input_ids[:, :tokens].contiguous()
    return selected, {
        "arrow_rows": len(texts),
        "articles": len(_article_hashes(texts)),
        "tokens_available": int(input_ids.shape[1]),
        "tokens_selected": tokens,
        "selected_token_sha256": sha256_bytes(selected.numpy().tobytes()),
        "tokenizer_class": type(tokenizer).__name__,
        "add_special_tokens": False,
        "join": "double-newline",
    }, _article_hashes(texts)


def _update_tail(record: dict[str, Any], rows: Any, scores: Any, torch: Any, limit: int) -> None:
    rows = rows.detach().to(device="cpu", dtype=torch.bfloat16)
    scores = scores.detach().to(device="cpu", dtype=torch.float32)
    if record["tail_rows"] is not None:
        rows = torch.cat((record["tail_rows"], rows), dim=0)
        scores = torch.cat((record["tail_scores"], scores), dim=0)
    keep = min(limit, int(scores.numel()))
    values, indexes = torch.topk(scores, keep, largest=True, sorted=True)
    record["tail_rows"] = rows.index_select(0, indexes).contiguous()
    record["tail_scores"] = values.contiguous()


def capture_samples(model: Any, calibration_sequences: Any, development_sequences: Any,
                    layers: Sequence[int], contract: Mapping[str, Any], events: Path, torch: Any):
    width = int(contract["model"]["ffn_intermediate_size"])
    sequence_length = int(contract["data"]["sequence_length"])
    cal_total = int(contract["data"]["calibration_tokens"])
    dev_total = int(contract["data"]["development_tokens"])
    cal_rows = int(contract["activation_sample"]["calibration_tail_stat_rows"])
    uniform_rows = int(contract["activation_sample"]["uniform_development_rows"])
    tail_rows = int(contract["activation_sample"]["tail_reservoir_rows"])
    if cal_total % cal_rows or dev_total % uniform_rows:
        raise ExperimentError("uniform sample sizes must divide their token streams")
    cal_stride = cal_total // cal_rows
    dev_stride = dev_total // uniform_rows
    records = {
        layer: {
            "calibration_sum_abs": torch.zeros(width, dtype=torch.float64),
            "calibration_tail_rows": [],
            "uniform_rows": [],
            "tail_rows": None,
            "tail_scores": None,
        }
        for layer in layers
    }
    state = {"split": "calibration", "offset": 0}
    handles = []

    def make_hook(layer: int):
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            value = inputs[0].detach().reshape(-1, width)
            if int(value.shape[0]) != sequence_length:
                raise ExperimentError("captured sequence activation row count differs")
            record = records[layer]
            if state["split"] == "calibration":
                record["calibration_sum_abs"] += value.float().abs().sum(dim=0).cpu().double()
                positions = torch.arange(
                    (-state["offset"]) % cal_stride,
                    sequence_length,
                    cal_stride,
                    device=value.device,
                )
                record["calibration_tail_rows"].append(
                    value.index_select(0, positions).to(device="cpu", dtype=torch.bfloat16)
                )
            else:
                positions = torch.arange(
                    (-state["offset"]) % dev_stride,
                    sequence_length,
                    dev_stride,
                    device=value.device,
                )
                record["uniform_rows"].append(
                    value.index_select(0, positions).to(device="cpu", dtype=torch.bfloat16)
                )
                scores = value.float().abs().amax(dim=1)
                keep = min(tail_rows, int(scores.numel()))
                local_scores, local_indexes = torch.topk(scores, keep, largest=True, sorted=True)
                _update_tail(
                    record,
                    value.index_select(0, local_indexes),
                    local_scores,
                    torch,
                    tail_rows,
                )
        return hook

    for layer in layers:
        module = model.model.layers[layer].mlp.down_proj
        if int(module.in_features) != width:
            raise ExperimentError(f"layer {layer} down-projection width differs")
        handles.append(module.register_forward_pre_hook(make_hook(layer)))
    try:
        for split, sequences in (
            ("calibration", calibration_sequences),
            ("development", development_sequences),
        ):
            state["split"] = split
            state["offset"] = 0
            for index, sequence in enumerate(sequences):
                append_event(events, "capture_sequence_started", split=split, sequence=index)
                with torch.inference_mode():
                    output = model.model(
                        input_ids=sequence.unsqueeze(0).to("cuda:0"),
                        use_cache=False,
                        return_dict=False,
                    )
                del output
                torch.cuda.synchronize()
                state["offset"] += sequence_length
                append_event(events, "capture_sequence_completed", split=split, sequence=index)
    finally:
        for handle in handles:
            handle.remove()
    finalized: dict[int, dict[str, Any]] = {}
    for layer, record in records.items():
        calibration_tail = torch.cat(record["calibration_tail_rows"], dim=0).contiguous()
        uniform = torch.cat(record["uniform_rows"], dim=0).contiguous()
        if tuple(calibration_tail.shape) != (cal_rows, width):
            raise ExperimentError(f"layer {layer} calibration statistic sample differs")
        if tuple(uniform.shape) != (uniform_rows, width):
            raise ExperimentError(f"layer {layer} uniform development sample differs")
        if record["tail_rows"] is None or tuple(record["tail_rows"].shape) != (tail_rows, width):
            raise ExperimentError(f"layer {layer} tail reservoir differs")
        channel_p99 = torch.quantile(
            calibration_tail.float().abs().to("cuda:0"), 0.99, dim=0
        ).cpu()
        finalized[layer] = {
            "calibration_mean_abs": (record["calibration_sum_abs"] / cal_total).float(),
            "calibration_p99_abs": channel_p99,
            "uniform": uniform,
            "tail": record["tail_rows"],
            "tail_scores": record["tail_scores"],
        }
        del calibration_tail
    return finalized


def _method_block_size(method: str) -> int | None:
    if method.endswith("32"):
        return 32
    if method.endswith("128"):
        return 128
    if method == "full_hadamard":
        return 12288
    return None


def evaluate_layer(model: Any, layer: int, sample: Mapping[str, Any], events: Path, torch: Any):
    mean_abs = sample["calibration_mean_abs"]
    channel_p99 = sample["calibration_p99_abs"]
    permutation_lists = {
        32: massdiff_permutation(mean_abs, channel_p99, 32),
        128: massdiff_permutation(mean_abs, channel_p99, 128),
    }
    permutations = {
        block: torch.tensor(values, device="cuda:0", dtype=torch.long)
        for block, values in permutation_lists.items()
    }
    representative_analysis: dict[str, object] = {}
    for block, permutation in permutation_lists.items():
        per_block = [permutation[begin:begin + block] for begin in range(0, len(permutation), block)]
        for representatives_per_block in (1, 2):
            representatives = [
                channel
                for current in per_block
                for channel in current[:representatives_per_block]
            ]
            key = f"b{block}_r{representatives_per_block}"
            representative_analysis[key] = {
                "channels_sha256": canonical_sha256(representatives),
                "uniform": representative_coverage(
                    sample["uniform"].to("cuda:0"), representatives, torch
                ),
                "tail_reservoir": representative_coverage(
                    sample["tail"].to("cuda:0"), representatives, torch
                ),
            }
    weight = model.model.layers[layer].mlp.down_proj.weight.detach().float()
    if tuple(weight.shape) != (4096, 12288):
        raise ExperimentError(f"layer {layer} down-projection weight shape differs: {tuple(weight.shape)}")
    samples = {
        "uniform": sample["uniform"].to(device="cuda:0", dtype=torch.float32),
        "tail_reservoir": sample["tail"].to(device="cuda:0", dtype=torch.float32),
    }
    references = {name: values @ weight.T for name, values in samples.items()}
    rows = []
    for method in METHOD_NAMES:
        append_event(events, "method_started", layer=layer, method=method)
        block = _method_block_size(method)
        permutation = permutations.get(block) if method.startswith("perq") else None
        transformed_weight = transform_tensor(weight, method, permutation, torch)
        reconstructed_weight, weight_quantization = quantize_weight_reconstruction(
            transformed_weight, torch
        )
        evaluations = {}
        for sample_name, activation in samples.items():
            transformed_activation = transform_tensor(activation, method, permutation, torch)
            reconstructed_activation, activation_quantization = quantize_activation_reconstruction(
                transformed_activation, torch
            )
            predicted = reconstructed_activation @ reconstructed_weight.T
            evaluations[sample_name] = {
                **output_error_metrics(predicted, references[sample_name], torch),
                "activation_quantization": activation_quantization,
                "post_transform_max_abs": float(transformed_activation.abs().amax().item()),
                "post_transform_rms": float(transformed_activation.square().mean().sqrt().item()),
            }
            del transformed_activation, reconstructed_activation, predicted
        row = {
            "schema_version": SCHEMA_VERSION,
            "layer": layer,
            "site": f"model.layers.{layer}.mlp.down_proj",
            "method": method,
            "b": block,
            "permutation": "massdiff_tail_order" if method.startswith("perq") else "none",
            "permutation_sha256": canonical_sha256(permutation_lists[block])
            if method.startswith("perq") else "not_applicable",
            "weight_quantization": weight_quantization,
            "uniform": evaluations["uniform"],
            "tail_reservoir": evaluations["tail_reservoir"],
            "execution_kind": "numerical_only",
        }
        rows.append(row)
        append_event(
            events,
            "method_completed",
            layer=layer,
            method=method,
            mean_output_error=row["uniform"]["mean_output_error"],
            tail_output_error=row["uniform"]["tail_output_error"],
        )
        del transformed_weight, reconstructed_weight
        torch.cuda.empty_cache()
    by_method = {row["method"]: row for row in rows}
    full_error = float(by_method["full_hadamard"]["uniform"]["mean_output_error"])
    identity_error = float(by_method["identity"]["uniform"]["mean_output_error"])
    gaps = {}
    for method in ("perq_h32", "perq_h128"):
        error = float(by_method[method]["uniform"]["mean_output_error"])
        denominator = identity_error - full_error
        gaps[method] = {
            "absolute_remaining_gap": error - full_error,
            "identity_to_full_improvement": denominator,
            "remaining_gap_fraction": (error - full_error) / denominator
            if denominator > 0 else "not_applicable",
        }
    return rows, representative_analysis, gaps, permutation_lists


def aggregate_results(rows: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    aggregates: dict[str, object] = {}
    for method in METHOD_NAMES:
        selected = [row for row in rows if row["method"] == method]
        if len(selected) != 6:
            raise ExperimentError(f"method {method} does not have six layer rows")
        numerator = sum(float(row["uniform"]["squared_error_sum"]) for row in selected)
        denominator = sum(float(row["uniform"]["reference_squared_sum"]) for row in selected)
        aggregates[method] = {
            "layers": [int(row["layer"]) for row in selected],
            "mean_output_error": numerator / denominator,
            "mean_layer_tail_output_error": statistics.fmean(
                float(row["uniform"]["tail_output_error"]) for row in selected
            ),
            "mean_tail_reservoir_output_error": statistics.fmean(
                float(row["tail_reservoir"]["mean_output_error"]) for row in selected
            ),
        }
    identity = float(aggregates["identity"]["mean_output_error"])
    full = float(aggregates["full_hadamard"]["mean_output_error"])
    denominator = identity - full
    remaining = {}
    for method in ("perq_h32", "perq_h128"):
        value = float(aggregates[method]["mean_output_error"])
        remaining[method] = {
            "absolute_remaining_gap": value - full,
            "identity_to_full_improvement": denominator,
            "remaining_gap_fraction": (value - full) / denominator
            if denominator > 0 else "not_applicable",
        }
    strongest = min(METHOD_NAMES, key=lambda name: float(aggregates[name]["mean_output_error"]))
    return {
        "by_method": aggregates,
        "remaining_quality_gap": remaining,
        "strongest_quality_control": strongest,
    }


def _environment(torch: Any, transformers: Any, pyarrow: Any) -> dict[str, object]:
    from importlib.metadata import version

    device = torch.cuda.get_device_properties(0)
    return {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "pyarrow": pyarrow.__version__,
        "safetensors": version("safetensors"),
        "cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_name": device.name,
        "device_uuid": str(getattr(device, "uuid", "unavailable")),
        "device_total_memory_bytes": int(device.total_memory),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
    }


def _verify_input_hashes(contract: Mapping[str, Any]) -> dict[str, object]:
    model = contract["model"]
    snapshot = Path(model["snapshot"]).resolve(strict=True)
    named = {
        "config.json": model["config_sha256"],
        "model.safetensors.index.json": model["index_sha256"],
        "tokenizer.json": model["tokenizer_sha256"],
        "tokenizer_config.json": model["tokenizer_config_sha256"],
    }
    checked = {}
    for name, expected in named.items():
        observed = sha256_file(snapshot / name)
        if observed != expected:
            raise ExperimentError(f"model metadata hash differs for {name}")
        checked[name] = observed
    shard_hashes = []
    for index, expected in enumerate(model["weight_shards_sha256"], start=1):
        path = snapshot / f"model-{index:05d}-of-00005.safetensors"
        observed = sha256_file(path)
        if observed != expected:
            raise ExperimentError(f"model shard hash differs for {path.name}")
        shard_hashes.append(observed)
    data = contract["data"]
    for role in ("calibration", "development"):
        path = Path(data[f"{role}_arrow"]).resolve(strict=True)
        observed = sha256_file(path)
        if observed != data[f"{role}_arrow_sha256"]:
            raise ExperimentError(f"{role} Arrow hash differs")
        checked[f"{role}_arrow"] = observed
    checked["weight_shards"] = shard_hashes
    return checked


def _save_activation_cache(result_directory: Path, layer: int, sample: Mapping[str, Any],
                           permutations: Mapping[int, Sequence[int]], torch: Any) -> dict[str, object]:
    directory = result_directory / "activation-sample"
    directory.mkdir(mode=0o700, exist_ok=True)
    path = directory / f"layer-{layer:02d}.pt"
    payload = {
        "schema_version": "structured-rotations-v2-gate-a-activation-sample-v1",
        "layer": layer,
        "site": f"model.layers.{layer}.mlp.down_proj",
        "calibration_mean_abs": sample["calibration_mean_abs"],
        "calibration_p99_abs": sample["calibration_p99_abs"],
        "uniform_development_rows": sample["uniform"],
        "tail_reservoir_rows": sample["tail"],
        "tail_reservoir_scores": sample["tail_scores"],
        "massdiff_permutations": {str(block): list(value) for block, value in permutations.items()},
    }
    with path.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, 0o600)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def make_ledger(rows: Sequence[Mapping[str, Any]], result: Mapping[str, Any],
                active_gpu_hours: float) -> list[dict[str, object]]:
    common = {
        "run_id": result["run_id"],
        "code_commit": result["stage"]["head"],
        "model_revision": result["model"]["revision"],
        "target_sites": "six fixed FFN down_proj inputs",
        "permutation_seed": 0,
        "calibration_hash": result["data"]["calibration"]["selected_token_sha256"],
        "weight_quantizer": "symmetric narrow INT4 RTN; per-output-channel 21-factor MSE clipping",
        "activation_quantizer": "dynamic per-token asymmetric INT4 RTN",
        "scale_groups": "weight per output channel; activation per token row",
        "clipping_budget": "weight factors 1.00..0.80 inclusive; activation observed min/max",
        "execution_kind": "numerical_only",
        "gpu_model": result["environment"]["device_name"],
        "gpu_count": 1,
        "workload": "1024 uniform rows plus independent 1024-row activation-tail reservoir per layer",
        "dev_nll": "not_run",
        "final_ppl_if_run": "not_run",
        "segment_ms": "not_run",
        "end_to_end_ms": "not_run",
        "timing_dispersion": "not_run",
        "metadata_bytes": "reported in activation_cache_manifest",
        "active_gpu_hours": active_gpu_hours,
        "allocated_gpu_hours": "pending_terminal_accounting",
        "status": "complete",
        "next_gate": "Decision A after separate A2",
    }
    ledger = []
    for row in rows:
        ledger.append({
            **common,
            "method": row["method"],
            "target_sites": row["site"],
            "transform_definition": row["method"],
            "b": row["b"] if row["b"] is not None else "not_applicable",
            "r": "not_run",
            "mean_output_error": row["uniform"]["mean_output_error"],
            "tail_output_error": row["uniform"]["tail_output_error"],
        })
    return ledger


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    contract_path = args.contract.resolve(strict=True)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    validate_contract(contract)
    retained = Path(contract["storage"]["retained_root"]).resolve()
    result_directory = args.result_directory.resolve()
    if result_directory.parent != retained / "runs":
        raise ExperimentError(f"result directory must be a direct child of {retained / 'runs'}")
    result_directory.mkdir(mode=0o700)
    events = result_directory / "events.jsonl"
    events.touch(mode=0o600, exist_ok=False)
    append_event(events, "run_started", argv=sys.argv)
    stage = verify_stage(contract_path, contract)
    if platform.node().lower().find("login") >= 0:
        raise ExperimentError("quality experiment cannot run on a login node")
    if not os.environ.get("SLURM_JOB_ID"):
        raise ExperimentError("quality experiment lacks a Slurm job identity")

    import pyarrow
    import torch
    import transformers

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ExperimentError("quality experiment requires exactly one visible CUDA device")
    torch.manual_seed(20260910)
    torch.cuda.manual_seed_all(20260910)
    torch.backends.cuda.matmul.allow_tf32 = False
    input_hashes = _verify_input_hashes(contract)
    environment = _environment(torch, transformers, pyarrow)
    write_json(result_directory / "environment.json", environment)
    append_event(events, "inputs_verified", input_hashes=input_hashes, environment=environment)

    snapshot = Path(contract["model"]["snapshot"])
    calibration_ids, calibration_meta, calibration_articles = tokenize_stream(
        snapshot,
        Path(contract["data"]["calibration_arrow"]),
        int(contract["data"]["calibration_tokens"]),
        transformers=transformers,
        pyarrow=pyarrow,
        torch=torch,
    )
    development_ids, development_meta, development_articles = tokenize_stream(
        snapshot,
        Path(contract["data"]["development_arrow"]),
        int(contract["data"]["development_tokens"]),
        transformers=transformers,
        pyarrow=pyarrow,
        torch=torch,
    )
    overlap = calibration_articles & development_articles
    if overlap:
        raise ExperimentError(f"calibration/development article hashes overlap: {len(overlap)}")
    sequence_length = int(contract["data"]["sequence_length"])
    calibration_sequences = calibration_ids.reshape(-1, sequence_length)
    development_sequences = development_ids.reshape(-1, sequence_length)
    data_record = {
        "calibration": calibration_meta,
        "development": development_meta,
        "exact_article_hash_overlap": 0,
    }
    write_json(result_directory / "data.json", data_record)
    append_event(events, "data_frozen", data=data_record)

    append_event(events, "model_load_started")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).eval().to("cuda:0")
    if type(model).__name__ != "Qwen3ForCausalLM":
        raise ExperimentError(f"loaded model class differs: {type(model).__name__}")
    if int(model.config.num_hidden_layers) != 36 or int(model.config.intermediate_size) != 12288:
        raise ExperimentError("loaded Qwen3 dimensions differ")
    append_event(events, "model_load_completed", model_class=type(model).__name__)

    layers = [int(layer) for layer in contract["model"]["sampled_layers"]]
    samples = capture_samples(
        model,
        calibration_sequences,
        development_sequences,
        layers,
        contract,
        events,
        torch,
    )
    rows: list[dict[str, object]] = []
    coverage: dict[str, object] = {}
    layer_gaps: dict[str, object] = {}
    cache_manifest = []
    for layer in layers:
        append_event(events, "layer_evaluation_started", layer=layer)
        layer_rows, layer_coverage, gaps, permutations = evaluate_layer(
            model, layer, samples[layer], events, torch
        )
        rows.extend(layer_rows)
        coverage[str(layer)] = layer_coverage
        layer_gaps[str(layer)] = gaps
        cache_manifest.append(
            _save_activation_cache(result_directory, layer, samples[layer], permutations, torch)
        )
        append_event(events, "layer_evaluation_completed", layer=layer, gaps=gaps)
        del samples[layer]
        torch.cuda.empty_cache()
    aggregate = aggregate_results(rows)
    active_gpu_hours = (time.monotonic() - started) / 3600.0
    result = {
        "schema_version": SCHEMA_VERSION,
        "run_id": f"slurm-{os.environ['SLURM_JOB_ID']}",
        "recorded_at_utc": utc_now(),
        "stage": stage,
        "contract_sha256": stage["contract_sha256"],
        "model": {
            "repository": contract["model"]["substitution"],
            "revision": contract["model"]["revision"],
            "substitution_reason": contract["model"]["substitution_reason"],
        },
        "data": data_record,
        "input_hashes": input_hashes,
        "environment": environment,
        "execution_kind": "numerical_only",
        "layers": layers,
        "rows": rows,
        "representative_channel_coverage": coverage,
        "per_layer_remaining_quality_gap": layer_gaps,
        "aggregate": aggregate,
        "activation_cache_manifest": cache_manifest,
        "active_gpu_hours": active_gpu_hours,
        "allocated_gpu_hours": "pending_terminal_accounting",
        "confounds": [
            "Numerical packed reconstruction only; no native low-bit consumer or timing was run.",
            "The permitted Qwen3-8B substitution changes model/runtime comparability with the historical Llama-3-8B evidence.",
            "Layer-output error on retained activation samples is not whole-model perplexity.",
        ],
        "next_experiment": contract["next_experiment"],
    }
    write_json(result_directory / "results.json", result)
    write_jsonl(result_directory / "ledger.jsonl", make_ledger(rows, result, active_gpu_hours))
    append_event(
        events,
        "run_completed",
        active_gpu_hours=active_gpu_hours,
        aggregate=aggregate,
    )
    manifest_paths = sorted(
        path for path in result_directory.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    manifest_lines = [
        f"{sha256_file(path)}  {path.relative_to(result_directory)}"
        for path in manifest_paths
    ]
    manifest = result_directory / "SHA256SUMS"
    descriptor = os.open(manifest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write("\n".join(manifest_lines) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--result-directory", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except Exception as exc:
        result_directory = args.result_directory.resolve()
        if result_directory.is_dir() and not (result_directory / "failure.json").exists():
            write_json(result_directory / "failure.json", {
                "schema_version": "structured-rotations-v2-gate-a-failure-v1",
                "recorded_at_utc": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
        print(f"FAILED {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps({
        "status": "complete",
        "run_id": result["run_id"],
        "result": str(args.result_directory.resolve() / "results.json"),
        "aggregate": result["aggregate"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
