#!/usr/bin/env python3
"""One-shot Phase B 32-site quality/latency map and six-site PPL validation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
from typing import Callable, Mapping, Sequence

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.structured_hadamard.phase_a.real_model import benchmark as phase_a_benchmark
from experiments.structured_hadamard.phase_a.real_model import runtime as phase_a_runtime

if __package__:
    from . import analysis
    from .reference import TRANSFORMS, block_transform_invariants
    from .runtime import make_transform, replace_all_projection_linears_for_policy, transform_semantics
    from .schema import ROW_SCHEMA_VERSION, validate_rows
else:
    from experiments.structured_hadamard.phase_b import analysis  # type: ignore[no-redef]
    from experiments.structured_hadamard.phase_b.reference import (  # type: ignore[no-redef]
        TRANSFORMS,
        block_transform_invariants,
    )
    from experiments.structured_hadamard.phase_b.runtime import (  # type: ignore[no-redef]
        make_transform,
        replace_all_projection_linears_for_policy,
        transform_semantics,
    )
    from experiments.structured_hadamard.phase_b.schema import (  # type: ignore[no-redef]
        ROW_SCHEMA_VERSION,
        validate_rows,
    )


SCHEMA_VERSION = "phase-b-map-result-v1"
SEED = 20260902
CALIBRATION_SEQUENCES = 16
CALIBRATION_TOKENS = 512
CALIBRATION_ROWS = CALIBRATION_SEQUENCES * CALIBRATION_TOKENS
CALIBRATION_SUBSETS = {
    analysis.SUBSET_NAMES[0]: (0, 8 * CALIBRATION_TOKENS),
    analysis.SUBSET_NAMES[1]: (8 * CALIBRATION_TOKENS, CALIBRATION_ROWS),
}
PHASE_A_HFULL_PPL = 101.04225822787963
PHASE_A_RESULT_SHA256 = "a173ee7fc4d43ab4b33123190ea4e7893c1d5bf421587f07e2502b75fcfd61d4"


class DriverError(RuntimeError):
    """The single Phase B scientific run is incomplete or incomparable."""


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_event(path: Path, event: str, *, announce: bool = False, **values: object) -> None:
    payload = {"recorded_at_utc": _utc(), "event": event, **values}
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
    with os.fdopen(descriptor, "ab") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    if announce:
        print("STATUS " + json.dumps(payload, sort_keys=True), flush=True)


def _exclusive_json(path: Path, value: object) -> None:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _exclusive_jsonl(path: Path, values: Sequence[object]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        for value in values:
            stream.write((json.dumps(value, sort_keys=True) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def _append_jsonl(path: Path, value: object) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
    with os.fdopen(descriptor, "ab") as stream:
        stream.write((json.dumps(value, sort_keys=True) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentiles(samples: Sequence[float]) -> dict[str, float | int]:
    if not samples or any(not math.isfinite(value) or value < 0 for value in samples):
        raise DriverError("timing samples must be finite and nonnegative")
    ordered = sorted(samples)
    selected = lambda fraction: ordered[round((len(ordered) - 1) * fraction)]
    return {
        "minimum_ms": ordered[0], "p10_ms": selected(0.10),
        "median_ms": statistics.median(ordered), "p90_ms": selected(0.90),
        "maximum_ms": ordered[-1], "mean_ms": statistics.fmean(ordered),
        "samples": len(ordered),
    }


def _cuda_benchmark(function: Callable[[], object], *, warmups: int, repetitions: int,
                    torch: object) -> tuple[dict[str, float | int], list[float]]:
    for _ in range(warmups):
        function()
    torch.cuda.synchronize()
    events = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        events.append((start, end))
    torch.cuda.synchronize()
    samples = [float(start.elapsed_time(end)) for start, end in events]
    return _percentiles(samples), samples


def _cleanup_cuda(torch: object) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _calibration_sequences(input_ids: object) -> object:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < CALIBRATION_ROWS:
        raise DriverError("calibration token stream is shorter than exactly 8192 sampled rows")
    return input_ids[:, :CALIBRATION_ROWS].reshape(CALIBRATION_SEQUENCES, CALIBRATION_TOKENS)


def _capture_site(model: object, sequences: object, *, layer: int, device: str,
                  torch: object) -> tuple[object, object]:
    module = model.model.layers[layer].mlp.down_proj
    captured_inputs, captured_outputs = [], []

    def hook(_module, inputs, output):
        captured_inputs.append(inputs[0].detach().reshape(-1, module.in_features).to(
            device="cpu", dtype=torch.float16,
        ))
        captured_outputs.append(output.detach().reshape(-1, module.out_features).to(
            device="cpu", dtype=torch.float16,
        ))

    handle = module.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            for sequence in sequences:
                model(input_ids=sequence.unsqueeze(0).to(device), use_cache=False)
        torch.cuda.synchronize()
    finally:
        handle.remove()
    inputs = torch.cat(captured_inputs, dim=0).contiguous()
    outputs = torch.cat(captured_outputs, dim=0).contiguous()
    if tuple(inputs.shape) != (CALIBRATION_ROWS, 14336):
        raise DriverError(f"captured site input shape differs: {tuple(inputs.shape)}")
    if tuple(outputs.shape) != (CALIBRATION_ROWS, 4096):
        raise DriverError(f"captured site reference shape differs: {tuple(outputs.shape)}")
    return inputs, outputs


def _cache_site(path: Path, *, layer: int, inputs: object, references: object, torch: object) -> dict[str, object]:
    payload = {
        "schema_version": "phase-b-site-cache-v1", "layer": layer,
        "site": f"model.layers.{layer}.mlp.down_proj",
        "sequence_count": CALIBRATION_SEQUENCES, "tokens_per_sequence": CALIBRATION_TOKENS,
        "sampled_token_rows": CALIBRATION_ROWS, "dtype": "float16",
        "inputs": inputs, "reference_outputs": references,
    }
    with path.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _quality_metrics(module: object, transform: object, inputs: object, references: object,
                     *, device: str, chunk_rows: int, torch: object) -> tuple[dict[str, object], dict[str, object]]:
    accumulators = {
        name: {"error": 0.0, "reference": 0.0, "maximum": 0.0, "square": 0.0, "elements": 0}
        for name in CALIBRATION_SUBSETS
    }
    with torch.inference_mode():
        for subset_name, (subset_begin, subset_end) in CALIBRATION_SUBSETS.items():
            for begin in range(subset_begin, subset_end, chunk_rows):
                end = min(begin + chunk_rows, subset_end)
                activation = inputs[begin:end].to(device)
                reference = references[begin:end].to(device)
                transformed = transform.online(activation)
                predicted = module(activation)
                difference = predicted.float() - reference.float()
                record = accumulators[subset_name]
                record["error"] += float(difference.square().sum().item())
                record["reference"] += float(reference.float().square().sum().item())
                record["maximum"] = max(record["maximum"], float(transformed.float().abs().amax().item()))
                record["square"] += float(transformed.float().square().sum().item())
                record["elements"] += int(transformed.numel())
                del activation, reference, transformed, predicted, difference
    quality_subsets, outlier_subsets = {}, {}
    for name, value in accumulators.items():
        if value["reference"] <= 0 or value["elements"] <= 0:
            raise DriverError("quality denominator or activation count is nonpositive")
        quality_subsets[name] = {
            "rows": CALIBRATION_SUBSETS[name][1] - CALIBRATION_SUBSETS[name][0],
            "squared_error_sum": value["error"], "reference_squared_sum": value["reference"],
            "normalized_output_error": value["error"] / value["reference"],
        }
        outlier_subsets[name] = {
            "rows": CALIBRATION_SUBSETS[name][1] - CALIBRATION_SUBSETS[name][0],
            "max_abs": value["maximum"], "rms": math.sqrt(value["square"] / value["elements"]),
        }
    error = sum(value["error"] for value in accumulators.values())
    reference = sum(value["reference"] for value in accumulators.values())
    square = sum(value["square"] for value in accumulators.values())
    elements = sum(value["elements"] for value in accumulators.values())
    quality = {
        "metric": "normalized_output_mse", "unit": "ratio",
        "definition": "sum((accepted_packed_W4A4_output - fp16_reference_output)^2) / sum(fp16_reference_output^2)",
        "full": {"rows": CALIBRATION_ROWS, "squared_error_sum": error,
                 "reference_squared_sum": reference, "normalized_output_error": error / reference},
        "subsets": quality_subsets,
    }
    outlier = {
        "statistic": "max absolute value and RMS after the online transform, before A4 quantization",
        "unit": "activation_value",
        "full": {"rows": CALIBRATION_ROWS,
                 "max_abs": max(value["maximum"] for value in accumulators.values()),
                 "rms": math.sqrt(square / elements)},
        "subsets": outlier_subsets,
    }
    return quality, outlier


def _map_site(original: object, inputs: object, references: object, *, layer: int,
              extension: object, outer: object, outer_order: int, args: argparse.Namespace,
              hardware: Mapping[str, object], torch: object) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    wrapper_class = phase_a_runtime._packed_linear_class(torch)
    rows, raw = [], []
    timing_activation = inputs[:1].to(args.device)
    for transform_name in TRANSFORMS:
        transform = make_transform(
            transform_name, original.in_features, torch_module=torch,
            full_outer_matrix=outer, full_outer_order=outer_order,
        )
        module = wrapper_class(original, extension, transform).eval()
        quality, outlier = _quality_metrics(
            module, transform, inputs, references, device=args.device,
            chunk_rows=args.proxy_chunk_rows, torch=torch,
        )
        if transform_name == "I":
            transform_samples = [0.0] * args.timing_repetitions
            transform_timing = _percentiles(transform_samples)
        else:
            transform_timing, transform_samples = _cuda_benchmark(
                lambda: transform.online(timing_activation), warmups=args.timing_warmups,
                repetitions=args.timing_repetitions, torch=torch,
            )
        layer_timing, layer_samples = _cuda_benchmark(
            lambda: module(timing_activation), warmups=args.timing_warmups,
            repetitions=args.timing_repetitions, torch=torch,
        )
        row = {
            "schema_version": ROW_SCHEMA_VERSION, "source_commit": args.source_commit,
            "layer": layer, "site": f"model.layers.{layer}.mlp.down_proj",
            "transform": transform_name,
            "model": {"repository": args.model_repository, "revision": args.model_revision},
            "data": {
                "dataset": "Salesforce/wikitext", "configuration": "wikitext-2-raw-v1",
                "revision": args.dataset_revision, "calibration_split": args.calibration_split,
                "calibration_sha256": args.calibration_sha256,
            },
            "quantization": {
                "weights": "signed symmetric int4 per output row",
                "activations": "signed symmetric dynamic int4 per token row",
                "accumulation": "int32", "runtime": "accepted packed W4A4 CUTLASS extension",
                "runtime_sha256": args.extension_sha256,
            },
            "calibration": {
                "sequences": CALIBRATION_SEQUENCES, "tokens_per_sequence": CALIBRATION_TOKENS,
                "sampled_token_rows": CALIBRATION_ROWS, "dtype": "float16", "expanded": False,
                "selection": "first 8192 tokens from the pinned calibration split, reshaped 16x512",
                "subsets": {name: {"begin_row": begin, "end_row": end}
                            for name, (begin, end) in CALIBRATION_SUBSETS.items()},
            },
            "shape": {"calibration_input": [8192, 14336], "timing_input": [1, 14336],
                      "reference_output": [8192, 4096]},
            "block_semantics": transform_semantics(
                transform_name, original.in_features, full_outer_order=outer_order,
            ),
            "quality": quality, "activation_outlier": outlier,
            "timing": {
                "unit": "milliseconds", "warmups": args.timing_warmups,
                "repetitions": args.timing_repetitions,
                "synchronization": "CUDA events with terminal torch.cuda.synchronize",
                "transform": transform_timing,
                "affected_packed_w4a4_layer": layer_timing,
            },
            "fusion": args.fusion, "seed": args.seed, "hardware": dict(hardware),
        }
        rows.append(row)
        raw.extend(
            {"layer": layer, "site": row["site"], "transform": transform_name,
             "metric": "transform", "repetition": index, "milliseconds": value}
            for index, value in enumerate(transform_samples)
        )
        raw.extend(
            {"layer": layer, "site": row["site"], "transform": transform_name,
             "metric": "affected_packed_w4a4_layer", "repetition": index,
             "milliseconds": value}
            for index, value in enumerate(layer_samples)
        )
        del module, transform
        _cleanup_cuda(torch)
    del timing_activation
    return rows, raw


def _policy_model(args: argparse.Namespace, *, extension: object, outer: object, outer_order: int,
                  cheaper_layer: int | None, torch: object):
    model = phase_a_runtime.load_cached_llama3(args.snapshot).eval().to(args.device)
    transforms = {
        layer: make_transform(
            "H32" if layer == cheaper_layer else "Hfull", 14336, torch_module=torch,
            full_outer_matrix=outer, full_outer_order=outer_order,
        )
        for layer in range(32)
    }
    replaced = replace_all_projection_linears_for_policy(
        model, extension=extension, down_transforms=transforms, torch_module=torch,
    )
    if len(replaced) != 224:
        raise DriverError("policy did not install exactly 224 accepted packed projections")
    return model


def _validate_selected_sites(args: argparse.Namespace, *, selection: Mapping[str, object],
                             evaluation_ids: object, extension: object, outer: object,
                             outer_order: int, torch: object) -> list[dict[str, object]]:
    validations = []
    validation_path = args.output_directory / "six-site-validation.jsonl"
    validation_path.touch(mode=0o600, exist_ok=False)
    groups = {
        **{int(layer): "most_sensitive" for layer in selection["most_sensitive"]},
        **{int(layer): "least_sensitive" for layer in selection["least_sensitive"]},
    }
    for layer in selection["selected_sites"]:
        layer = int(layer)
        _append_event(args.status_path, "ppl_validation_started", layer=layer, group=groups[layer])
        model = _policy_model(
            args, extension=extension, outer=outer, outer_order=outer_order,
            cheaper_layer=layer, torch=torch,
        )
        quality = phase_a_benchmark.evaluate_perplexity(
            model, evaluation_ids, sequence_length=args.ppl_sequence_length,
            device=args.device, torch=torch,
        )
        record = {
            "schema_version": "phase-b-six-site-validation-row-v1", "layer": layer,
            "site": f"model.layers.{layer}.mlp.down_proj", "predicted_group": groups[layer],
            "frozen_proxy": analysis.PRIMARY_PROXY, "replacement": "Hfull -> H32 at this site only",
            "all_other_down_projection_choices": "Hfull", "all_other_settings": "accepted Phase A",
            "perplexity": quality["perplexity"], "ppl_impact_vs_phase_a_hfull":
                float(quality["perplexity"]) - PHASE_A_HFULL_PPL,
            "quality": quality,
        }
        _append_jsonl(validation_path, record)
        validations.append(record)
        _append_event(
            args.status_path, "ppl_validation_completed", layer=layer,
            perplexity=record["perplexity"], ppl_impact=record["ppl_impact_vs_phase_a_hfull"],
        )
        del model
        _cleanup_cuda(torch)
    return validations


def _sequence_nll(model: object, sequences: object, *, indices: Sequence[int], device: str,
                  torch: object) -> float:
    total_nll, total_tokens = 0.0, 0
    model.config.use_cache = False
    with torch.inference_mode():
        for index in indices:
            batch = sequences[index].unsqueeze(0).to(device)
            logits = model(input_ids=batch, use_cache=False).logits
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
                batch[:, 1:].reshape(-1), reduction="sum",
            )
            total_nll += float(loss.item())
            total_tokens += int(batch[:, 1:].numel())
            del batch, logits, loss
    torch.cuda.synchronize()
    return total_nll / total_tokens


def _stronger_proxy(args: argparse.Namespace, *, calibration_sequences: object,
                    extension: object, outer: object, outer_order: int,
                    torch: object) -> dict[str, object]:
    """The only predeclared proxy switch: two disjoint short-sequence NLLs."""

    subset_indices = {analysis.SUBSET_NAMES[0]: (0,), analysis.SUBSET_NAMES[1]: (8,)}
    baseline_model = _policy_model(
        args, extension=extension, outer=outer, outer_order=outer_order,
        cheaper_layer=None, torch=torch,
    )
    baseline = {
        name: _sequence_nll(baseline_model, calibration_sequences, indices=indices,
                            device=args.device, torch=torch)
        for name, indices in subset_indices.items()
    }
    del baseline_model
    _cleanup_cuda(torch)
    subset_scores = {name: {} for name in subset_indices}
    rows = []
    for layer in range(32):
        model = _policy_model(
            args, extension=extension, outer=outer, outer_order=outer_order,
            cheaper_layer=layer, torch=torch,
        )
        layer_scores = {}
        for name, indices in subset_indices.items():
            nll = _sequence_nll(
                model, calibration_sequences, indices=indices, device=args.device, torch=torch,
            )
            layer_scores[name] = nll - baseline[name]
            subset_scores[name][layer] = layer_scores[name]
        rows.append({
            "layer": layer, "site": f"model.layers.{layer}.mlp.down_proj",
            "proxy": analysis.STRONGER_PROXY, "replacement": "Hfull -> H32",
            "subset_nll_impacts": layer_scores,
        })
        del model
        _cleanup_cuda(torch)
    scores = {
        layer: statistics.fmean(subset_scores[name][layer] for name in subset_indices)
        for layer in range(32)
    }
    rho = analysis.spearman_rho(
        subset_scores[analysis.SUBSET_NAMES[0]], subset_scores[analysis.SUBSET_NAMES[1]],
    )
    return {
        "schema_version": "phase-b-proxy-switch-v1",
        "reason": "primary local-error ranking failed its predeclared six-site PPL agreement test",
        "switch_count": 1, "proxy": analysis.STRONGER_PROXY,
        "definition": "end-to-end mean token NLL impact of one Hfull-to-H32 site replacement",
        "subset_sequence_indices": {name: list(indices) for name, indices in subset_indices.items()},
        "baseline_hfull_nll": baseline, "rows": rows,
        "scores": {str(layer): scores[layer] for layer in range(32)},
        "stability": {"statistic": analysis.STABILITY_STATISTIC, "spearman_rho": rho,
                      "threshold": analysis.STABILITY_THRESHOLD,
                      "stable": rho >= analysis.STABILITY_THRESHOLD},
    }


def _verify_phase_a_result(args: argparse.Namespace) -> None:
    if args.phase_a_results_sha256 != PHASE_A_RESULT_SHA256:
        raise DriverError("Phase A result digest argument differs from the accepted digest")
    if _sha256(args.phase_a_results) != PHASE_A_RESULT_SHA256:
        raise DriverError("accepted Phase A results bytes differ")
    value = json.loads(args.phase_a_results.read_text(encoding="utf-8"))
    matches = [row for row in value["variants"] if row["variant"] == "w4a4_hfull_down_proj"]
    if len(matches) != 1 or float(matches[0]["quality"]["perplexity"]) != PHASE_A_HFULL_PPL:
        raise DriverError("accepted Phase A Hfull PPL baseline differs")


def run(args: argparse.Namespace) -> dict[str, object]:
    import torch

    if "login" in platform.node().lower():
        raise DriverError(f"scientific payload cannot run on login host {platform.node()}")
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_JOB_PARTITION"):
        raise DriverError("scientific payload lacks Slurm allocation identity")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise DriverError("scientific payload requires exactly one visible CUDA device")
    capability = tuple(torch.cuda.get_device_capability(0))
    if capability not in ((8, 0), (8, 6)):
        raise DriverError(f"accepted extension requires native SM80/SM86, observed {capability}")
    if args.model_repository != phase_a_runtime.MODEL_REPOSITORY or args.model_revision != phase_a_runtime.MODEL_REVISION:
        raise DriverError("model identity differs from accepted Phase A")
    if args.seed != SEED or args.fusion != "none":
        raise DriverError("seed or sequential fusion contract differs")
    if args.timing_warmups != 20 or args.timing_repetitions != 100:
        raise DriverError("timing contract differs from 20 warmups and 100 repetitions")
    if args.calibration_sequences != 16 or args.calibration_tokens != 512 or args.expand_calibration:
        raise DriverError("calibration must remain exactly 16x512 without expansion")
    if _sha256(args.calibration_arrow) != args.calibration_sha256:
        raise DriverError("calibration Arrow digest differs")
    if _sha256(args.evaluation_arrow) != args.evaluation_sha256:
        raise DriverError("evaluation Arrow digest differs")
    if _sha256(args.extension) != args.extension_sha256:
        raise DriverError("accepted packed-W4A4 extension digest differs")
    _verify_phase_a_result(args)
    block_checks = {str(block): block_transform_invariants(block) for block in (32, 128)}
    if any(max(values.values()) > args.correctness_tolerance for values in block_checks.values()):
        raise DriverError(f"block transform/folding invariant failed: {block_checks}")
    full_check = phase_a_runtime.unquantized_one_block_smoke()
    if max(full_check.values()) > args.correctness_tolerance:
        raise DriverError(f"inherited Hfull invariant failed: {full_check}")

    args.output_directory.mkdir(mode=0o700)
    cache_directory = args.output_directory / "site-cache"
    cache_directory.mkdir(mode=0o700)
    _append_event(args.status_path, "scientific_driver_started", announce=True,
                  output_directory=str(args.output_directory))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    _, calibration_ids, calibration_preprocessing = phase_a_benchmark._tokenize_wikitext(
        args.snapshot, args.calibration_arrow,
    )
    _, evaluation_ids, evaluation_preprocessing = phase_a_benchmark._tokenize_wikitext(
        args.snapshot, args.evaluation_arrow,
    )
    calibration_sequences = _calibration_sequences(calibration_ids)

    gpu = torch.cuda.get_device_properties(0)
    hardware = {
        "name": gpu.name, "uuid": str(getattr(gpu, "uuid", "unavailable")),
        "total_memory_bytes": int(gpu.total_memory), "compute_capability": list(capability),
        "driver": torch._C._cuda_getDriverVersion() if hasattr(torch._C, "_cuda_getDriverVersion") else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "job_id": os.environ["SLURM_JOB_ID"], "partition": os.environ["SLURM_JOB_PARTITION"],
        "hostname": platform.node(),
    }
    _append_event(args.status_path, "gpu_visible", announce=True, hardware=hardware)
    extension = phase_a_runtime.load_extension(args.extension, args.extension_sha256)
    outer, outer_order = phase_a_runtime.load_quarot_outer_matrix(args.quarot_root, 14336, torch)
    outer = outer.to(device=args.device, dtype=torch.float32)

    model = phase_a_runtime.load_cached_llama3(args.snapshot).eval().to(args.device)
    map_rows, raw_timings, cache_manifest = [], [], []
    for layer in range(32):
        _append_event(args.status_path, "site_started", layer=layer)
        original = model.model.layers[layer].mlp.down_proj
        inputs, references = _capture_site(
            model, calibration_sequences, layer=layer, device=args.device, torch=torch,
        )
        cache_record = _cache_site(
            cache_directory / f"layer-{layer:02d}-down-proj.pt", layer=layer,
            inputs=inputs, references=references, torch=torch,
        )
        cache_manifest.append(cache_record)
        rows, raw = _map_site(
            original, inputs, references, layer=layer, extension=extension,
            outer=outer, outer_order=outer_order, args=args, hardware=hardware, torch=torch,
        )
        map_rows.extend(rows)
        raw_timings.extend(raw)
        _append_event(args.status_path, "site_completed", layer=layer,
                      cache_sha256=cache_record["sha256"])
        del inputs, references, rows, raw, original
        _cleanup_cuda(torch)
    del model
    _cleanup_cuda(torch)
    validate_rows(map_rows)
    map_path = args.output_directory / "map-rows.jsonl"
    raw_path = args.output_directory / "raw-timings.jsonl"
    cache_manifest_path = args.output_directory / "site-cache-manifest.json"
    _exclusive_jsonl(map_path, map_rows)
    _exclusive_jsonl(raw_path, raw_timings)
    _exclusive_json(cache_manifest_path, {
        "schema_version": "phase-b-site-cache-manifest-v1", "entries": cache_manifest,
        "total_bytes": sum(int(record["bytes"]) for record in cache_manifest),
    })

    selection = analysis.freeze_selection(map_rows)
    selection_path = args.output_directory / "selection-freeze.json"
    _exclusive_json(selection_path, selection)
    _append_event(args.status_path, "selection_frozen", selected_sites=selection["selected_sites"],
                  stability=selection["stability"])
    validations = _validate_selected_sites(
        args, selection=selection, evaluation_ids=evaluation_ids,
        extension=extension, outer=outer, outer_order=outer_order, torch=torch,
    )
    primary_agreement = analysis.proxy_agreement(selection, validations)
    active_scores = {int(layer): float(value) for layer, value in selection["scores"].items()}
    active_stability = float(selection["stability"]["spearman_rho"])
    proxy_record: dict[str, object] = {
        "primary": primary_agreement, "active_proxy": analysis.PRIMARY_PROXY,
        "switch_count": 0, "stronger": None,
    }
    proxy_validated = bool(primary_agreement["agrees"])
    if not proxy_validated:
        switched = _stronger_proxy(
            args, calibration_sequences=calibration_sequences, extension=extension,
            outer=outer, outer_order=outer_order, torch=torch,
        )
        switched_path = args.output_directory / "proxy-switch.json"
        _exclusive_json(switched_path, switched)
        active_scores = {int(layer): float(value) for layer, value in switched["scores"].items()}
        active_stability = float(switched["stability"]["spearman_rho"])
        stronger_agreement = analysis.proxy_agreement(
            selection, validations, scores=active_scores,
        )
        proxy_validated = bool(stronger_agreement["agrees"])
        proxy_record.update({
            "active_proxy": analysis.STRONGER_PROXY, "switch_count": 1,
            "stronger": {"agreement": stronger_agreement,
                         "artifact": str(switched_path),
                         "stability": switched["stability"]},
        })

    decision = analysis.literal_decision(
        map_rows, active_scores, stability_rho=active_stability,
        proxy_validated=proxy_validated,
    )
    table_path = args.output_directory / "phase-b-table.csv"
    figure_path = args.output_directory / "phase-b-quality-latency-map.svg"
    analysis.write_table(table_path, map_rows)
    analysis.write_figure(figure_path, map_rows)
    analysis_path = args.output_directory / "ranking-stability-analysis.json"
    _exclusive_json(analysis_path, {
        "schema_version": "phase-b-ranking-analysis-v1",
        "predeclared_primary_proxy": analysis.PRIMARY_PROXY,
        "predeclared_stronger_proxy": analysis.STRONGER_PROXY,
        "selection": selection, "proxy_validation": proxy_record,
        "active_scores": {str(layer): active_scores[layer] for layer in range(32)},
        "decision": decision,
    })
    key_paths = [map_path, raw_path, cache_manifest_path, selection_path,
                 args.output_directory / "six-site-validation.jsonl", table_path,
                 figure_path, analysis_path]
    artifact_hashes = {path.name: _sha256(path) for path in key_paths}
    result = {
        "schema_version": SCHEMA_VERSION, "recorded_at_utc": _utc(),
        "source_commit": args.source_commit, "phase_a_results_sha256": args.phase_a_results_sha256,
        "model": {"repository": args.model_repository, "revision": args.model_revision},
        "data": {"revision": args.dataset_revision,
                 "calibration_arrow": str(args.calibration_arrow),
                 "calibration_sha256": args.calibration_sha256,
                 "evaluation_arrow": str(args.evaluation_arrow),
                 "evaluation_sha256": args.evaluation_sha256},
        "preprocessing": {"calibration": calibration_preprocessing,
                          "evaluation": evaluation_preprocessing},
        "hardware": hardware, "correctness": {"H32": block_checks["32"],
                                               "H128": block_checks["128"],
                                               "Hfull": full_check},
        "B1_cache": {"sequences": 16, "tokens_per_sequence": 512,
                     "rows_per_site": 8192, "sites": 32,
                     "cache_manifest": str(cache_manifest_path), "expanded": False},
        "B2_map": {"rows": len(map_rows), "transforms": list(TRANSFORMS),
                   "fusion": "none", "map_rows": str(map_path),
                   "table": str(table_path), "figure": str(figure_path)},
        "B3_validation": {"selection_freeze": str(selection_path), "sites": 6,
                          "cheaper_transform": analysis.CHEAPER_TRANSFORM,
                          "phase_a_hfull_ppl": PHASE_A_HFULL_PPL,
                          "rows": str(args.output_directory / "six-site-validation.jsonl")},
        "proxy_validation": proxy_record, "literal_decision_B": decision,
        "phase_c_entered": False, "artifact_hashes": artifact_hashes,
    }
    _exclusive_json(args.output_directory / "results.json", result)
    _append_event(args.status_path, "scientific_driver_completed", announce=True,
                  decision=decision["conclusion"], proxy=proxy_record["active_proxy"])
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--model-repository", default=phase_a_runtime.MODEL_REPOSITORY)
    parser.add_argument("--model-revision", default=phase_a_runtime.MODEL_REVISION)
    parser.add_argument("--calibration-arrow", type=Path, required=True)
    parser.add_argument("--calibration-split", default="train")
    parser.add_argument("--calibration-sha256", required=True)
    parser.add_argument("--evaluation-arrow", type=Path, required=True)
    parser.add_argument("--evaluation-sha256", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--extension-sha256", required=True)
    parser.add_argument("--quarot-root", type=Path, required=True)
    parser.add_argument("--phase-a-results", type=Path, required=True)
    parser.add_argument("--phase-a-results-sha256", default=PHASE_A_RESULT_SHA256)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--calibration-sequences", type=int, default=CALIBRATION_SEQUENCES)
    parser.add_argument("--calibration-tokens", type=int, default=CALIBRATION_TOKENS)
    parser.add_argument("--expand-calibration", action="store_true")
    parser.add_argument("--proxy-chunk-rows", type=int, default=256)
    parser.add_argument("--ppl-sequence-length", type=int, default=1024)
    parser.add_argument("--timing-warmups", type=int, default=20)
    parser.add_argument("--timing-repetitions", type=int, default=100)
    parser.add_argument("--correctness-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--fusion", choices=("none",), default="none")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run(args)
    except BaseException as error:
        try:
            _append_event(args.status_path, "runtime_error", announce=True,
                          error=f"{type(error).__name__}: {error}")
        except BaseException:
            pass
        raise
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
