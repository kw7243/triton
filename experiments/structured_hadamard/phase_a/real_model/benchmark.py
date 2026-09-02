"""One-shot real-model Phase A quality and latency comparison."""

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
import time
from typing import Callable, Sequence

if __package__:
    from .runtime import (
        MODEL_REPOSITORY,
        MODEL_REVISION,
        IdentityTransform,
        TorchFullHadamard,
        load_cached_llama3,
        load_extension,
        load_quarot_outer_matrix,
        replace_all_projection_linears,
        sha256_file,
        unquantized_one_block_smoke,
    )
else:
    from runtime import (  # type: ignore[no-redef]
        MODEL_REPOSITORY,
        MODEL_REVISION,
        IdentityTransform,
        TorchFullHadamard,
        load_cached_llama3,
        load_extension,
        load_quarot_outer_matrix,
        replace_all_projection_linears,
        sha256_file,
        unquantized_one_block_smoke,
    )


SCHEMA_VERSION = "phase-a-real-model-result-v1"
VARIANTS = ("fp16", "w4a4_identity", "w4a4_hfull_down_proj")


class BenchmarkError(RuntimeError):
    """The scientific run is not complete and comparable."""


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_event(path: Path, event: str, **values: object) -> None:
    payload = {"recorded_at_utc": _utc(), "event": event, **values}
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
    with os.fdopen(descriptor, "ab") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _exclusive_json(path: Path, value: object) -> None:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _exclusive_jsonl(path: Path, values: Sequence[object]) -> None:
    encoded = b"".join((json.dumps(value, sort_keys=True) + "\n").encode("utf-8") for value in values)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _distribution_version(name: str) -> str:
    from importlib.metadata import version

    return version(name)


def _read_wikitext_arrow(path: Path) -> list[str]:
    import pyarrow as pa

    path = path.resolve(strict=True)
    with pa.memory_map(str(path), "r") as source:
        table = pa.ipc.open_stream(source).read_all()
    if table.column_names != ["text"]:
        raise BenchmarkError(f"WikiText Arrow columns differ: {table.column_names!r}")
    texts = table.column("text").to_pylist()
    if not texts or any(not isinstance(text, str) for text in texts):
        raise BenchmarkError("WikiText Arrow text column is empty or malformed")
    return texts


def _tokenize_wikitext(snapshot: Path, arrow_path: Path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True,
    )
    texts = _read_wikitext_arrow(arrow_path)
    joined = "\n\n".join(texts)
    encoded = tokenizer(joined, return_tensors="pt")
    input_ids = encoded.input_ids.cpu()
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 2:
        raise BenchmarkError(f"tokenized WikiText shape differs: {tuple(input_ids.shape)}")
    return tokenizer, input_ids, {
        "rows": len(texts),
        "join": "double-newline",
        "add_special_tokens": "tokenizer-default",
        "tokens": int(input_ids.numel()),
        "token_sha256": hashlib.sha256(input_ids.numpy().tobytes()).hexdigest(),
        "tokenizer_class": type(tokenizer).__name__,
    }


def _percentiles(samples: Sequence[float]) -> dict[str, float]:
    if not samples or any(not math.isfinite(value) or value < 0 for value in samples):
        raise BenchmarkError("timing samples must be nonnegative and finite")
    ordered = sorted(samples)

    def selected(fraction: float) -> float:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "minimum_ms": ordered[0],
        "p10_ms": selected(0.10),
        "median_ms": statistics.median(ordered),
        "p90_ms": selected(0.90),
        "maximum_ms": ordered[-1],
        "mean_ms": statistics.fmean(ordered),
        "samples": len(ordered),
    }


def _cuda_event_benchmark(function: Callable[[], object], *, warmups: int,
                          repetitions: int, torch: object) -> tuple[dict[str, float], list[float]]:
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


def evaluate_perplexity(model: object, input_ids: object, *, sequence_length: int,
                        device: str, torch: object) -> dict[str, object]:
    segments = int(input_ids.numel()) // sequence_length
    if segments <= 0:
        raise BenchmarkError("WikiText token stream is shorter than one PPL segment")
    total_nll = 0.0
    total_tokens = 0
    started = time.perf_counter()
    model.config.use_cache = False
    with torch.inference_mode():
        for index in range(segments):
            begin = index * sequence_length
            batch = input_ids[:, begin:begin + sequence_length].to(device)
            logits = model(input_ids=batch, use_cache=False).logits
            shift_logits = logits[:, :-1, :].float().reshape(-1, logits.shape[-1])
            labels = batch[:, 1:].reshape(-1)
            loss = torch.nn.functional.cross_entropy(shift_logits, labels, reduction="sum")
            total_nll += float(loss.item())
            total_tokens += int(labels.numel())
            del batch, logits, shift_logits, labels, loss
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    perplexity = math.exp(total_nll / total_tokens)
    if not math.isfinite(perplexity):
        raise BenchmarkError("perplexity is not finite")
    return {
        "dataset": "Salesforce/wikitext",
        "configuration": "wikitext-2-raw-v1",
        "split": "test",
        "sequence_length": sequence_length,
        "segments": segments,
        "evaluated_tokens": total_tokens,
        "tail_tokens_truncated": int(input_ids.numel()) - segments * sequence_length,
        "negative_log_likelihood_sum": total_nll,
        "perplexity": perplexity,
        "elapsed_seconds": elapsed,
    }


def _greedy_decode(model: object, prompt: object, *, output_length: int, torch: object):
    output = model(input_ids=prompt, use_cache=True)
    next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_token]
    cache = output.past_key_values
    for _ in range(output_length - 1):
        output = model(input_ids=next_token, past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token)
    return torch.cat(generated, dim=1)


def benchmark_decode(model: object, prompt: object, *, output_length: int, warmups: int,
                     repetitions: int, torch: object) -> tuple[dict[str, object], list[float]]:
    model.config.use_cache = True
    with torch.inference_mode():
        for _ in range(warmups):
            _greedy_decode(model, prompt, output_length=output_length, torch=torch)
        torch.cuda.synchronize()
        samples = []
        token_hash = None
        for _ in range(repetitions):
            torch.cuda.synchronize()
            started = time.perf_counter()
            generated = _greedy_decode(model, prompt, output_length=output_length, torch=torch)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - started) * 1000.0)
            current_hash = hashlib.sha256(generated.cpu().numpy().tobytes()).hexdigest()
            if token_hash is not None and current_hash != token_hash:
                raise BenchmarkError("greedy decode output changed across repetitions")
            token_hash = current_hash
    summary = _percentiles(samples)
    summary.update({
        "prompt_length": int(prompt.shape[1]),
        "output_length": output_length,
        "batch_size": int(prompt.shape[0]),
        "warmups": warmups,
        "repetitions": repetitions,
        "synchronization": "torch.cuda.synchronize before and after each repetition",
        "generation": "manual greedy argmax; use_cache=true; exact output length",
        "generated_token_sha256": token_hash,
        "median_ms_per_output_token": summary["median_ms"] / output_length,
        "median_tokens_per_second": output_length * 1000.0 / summary["median_ms"],
    })
    return summary, samples


def capture_down_projection_input(model: object, prompt: object, *, torch: object):
    module = model.model.layers[0].mlp.down_proj
    captured = []

    def hook(_module, inputs):
        captured.append(inputs[0][:, -1:, :].reshape(-1, inputs[0].shape[-1]).detach())

    handle = module.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            model(input_ids=prompt, use_cache=False)
        torch.cuda.synchronize()
    finally:
        handle.remove()
    if len(captured) != 1 or tuple(captured[0].shape) != (1, module.in_features):
        raise BenchmarkError("could not capture one real layer-0 down-projection decode row")
    return captured[0].contiguous()


def benchmark_affected_layer(model: object, prompt: object, *, transform: str,
                             warmups: int, repetitions: int, torch: object):
    module = model.model.layers[0].mlp.down_proj
    activation = capture_down_projection_input(model, prompt, torch=torch)
    layer_summary, layer_samples = _cuda_event_benchmark(
        lambda: module(activation), warmups=warmups, repetitions=repetitions, torch=torch,
    )
    if transform == "I":
        if not isinstance(module.transform, IdentityTransform):
            raise BenchmarkError("identity W4A4 layer does not use the host-alias identity")
        rotation_summary = {
            "minimum_ms": 0.0, "p10_ms": 0.0, "median_ms": 0.0,
            "p90_ms": 0.0, "maximum_ms": 0.0, "mean_ms": 0.0,
            "samples": repetitions,
        }
        rotation_samples = [0.0] * repetitions
    else:
        rotation_summary, rotation_samples = _cuda_event_benchmark(
            lambda: module.transform.online(activation), warmups=warmups,
            repetitions=repetitions, torch=torch,
        )
    summary = {
        "layer": 0,
        "site": "model.layers.0.mlp.down_proj",
        "input_shape": list(activation.shape),
        "transform": transform,
        "quantization": "packed signed W4A4; per-output-row W4; dynamic-per-token-row A4",
        "fusion": "none",
        "warmups": warmups,
        "repetitions": repetitions,
        "synchronization": "CUDA events with terminal torch.cuda.synchronize",
        "rotation": rotation_summary,
        "affected_quantized_layer": layer_summary,
    }
    raw = [
        {"metric": "rotation", "transform": transform, "repetition": index, "milliseconds": value}
        for index, value in enumerate(rotation_samples)
    ] + [
        {"metric": "affected_quantized_layer", "transform": transform,
         "repetition": index, "milliseconds": value}
        for index, value in enumerate(layer_samples)
    ]
    return summary, raw


def _load_model(snapshot: Path, *, device: str):
    model = load_cached_llama3(snapshot)
    model.eval()
    return model.to(device)


def _cleanup_cuda(torch: object) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _variant_record(variant: str, *, model: object, input_ids: object, prompt: object,
                    args: argparse.Namespace, torch: object) -> tuple[dict[str, object], list[dict[str, object]]]:
    _append_event(args.status_path, "variant_started", variant=variant)
    quality = evaluate_perplexity(
        model, input_ids, sequence_length=args.ppl_sequence_length,
        device=args.device, torch=torch,
    )
    decode, decode_samples = benchmark_decode(
        model, prompt, output_length=args.output_length, warmups=args.decode_warmups,
        repetitions=args.decode_repetitions, torch=torch,
    )
    record = {"variant": variant, "quality": quality, "decode": decode}
    raw = [
        {"metric": "end_to_end_decode", "variant": variant,
         "repetition": index, "milliseconds": value}
        for index, value in enumerate(decode_samples)
    ]
    _append_event(
        args.status_path, "variant_completed", variant=variant,
        perplexity=quality["perplexity"], decode_median_ms=decode["median_ms"],
    )
    return record, raw


def _overhead(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        raise BenchmarkError("overhead denominator must be positive")
    return (numerator / denominator - 1.0) * 100.0


def _decision(identity: MappingLike, hfull: MappingLike) -> dict[str, object]:
    end_to_end = _overhead(
        float(hfull["decode"]["median_ms_per_output_token"]),
        float(identity["decode"]["median_ms_per_output_token"]),
    )
    kernel = _overhead(
        float(hfull["layer"]["affected_quantized_layer"]["median_ms"]),
        float(identity["layer"]["affected_quantized_layer"]["median_ms"]),
    )
    if end_to_end >= 3.0 or kernel >= 5.0:
        conclusion = "A1: proceed to Phase B"
    elif end_to_end < 2.0 and kernel > 0.0:
        conclusion = (
            "A2: end-to-end overhead <2% but kernel overhead is visible; only the two "
            "plan-defined stress regimes may be proposed later and were not run here"
        )
    else:
        conclusion = "A1 not met and A2 not established by the literal thresholds"
    return {
        "hfull_vs_identity_end_to_end_overhead_percent": end_to_end,
        "hfull_vs_identity_affected_kernel_overhead_percent": kernel,
        "A1": "online transform overhead >=3% end-to-end or >=5% in an important affected kernel: proceed to Phase B",
        "A2": "end-to-end overhead <2% but kernel overhead visible: only two plan-defined later stress regimes may be proposed",
        "A3": "not established; this run alone cannot establish A3",
        "conclusion": conclusion,
    }


# A small structural alias keeps type checkers out of this executable's runtime.
MappingLike = dict[str, object]


def run(args: argparse.Namespace) -> dict[str, object]:
    import torch

    if platform.node().lower().find("login") >= 0:
        raise BenchmarkError(f"scientific payload cannot run on a login host: {platform.node()}")
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_JOB_PARTITION"):
        raise BenchmarkError("scientific payload lacks Slurm allocation identity")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise BenchmarkError("scientific payload requires exactly one visible CUDA device")
    capability = tuple(torch.cuda.get_device_capability(0))
    if capability not in ((8, 0), (8, 6)):
        raise BenchmarkError(f"accepted extension requires native SM80/SM86, observed SM{capability}")
    if args.model_revision != MODEL_REVISION or args.model_repository != MODEL_REPOSITORY:
        raise BenchmarkError("model identity differs from the accepted Llama-3 cache")
    if args.fusion != "none":
        raise BenchmarkError("sequential transform and W4A4 must record fusion=none")
    if sha256_file(args.dataset_arrow) != args.dataset_sha256:
        raise BenchmarkError("WikiText-2 Arrow digest differs")
    if sha256_file(args.extension) != args.extension_sha256:
        raise BenchmarkError("packed W4A4 extension digest differs")
    smoke = unquantized_one_block_smoke()
    if max(smoke.values()) > args.correctness_tolerance:
        raise BenchmarkError(f"unquantized one-block transform smoke failed: {smoke!r}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tokenizer, input_ids, preprocessing = _tokenize_wikitext(args.snapshot, args.dataset_arrow)
    if input_ids.shape[1] < args.prompt_length:
        raise BenchmarkError("WikiText token stream is shorter than the decode prompt")
    prompt = input_ids[:, :args.prompt_length].to(args.device)

    gpu = torch.cuda.get_device_properties(0)
    gpu_record = {
        "name": gpu.name,
        "uuid": str(getattr(gpu, "uuid", "unavailable")),
        "total_memory_bytes": int(gpu.total_memory),
        "compute_capability": list(capability),
        "driver": torch._C._cuda_getDriverVersion() if hasattr(torch._C, "_cuda_getDriverVersion") else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    _append_event(
        args.status_path, "gpu_visible", job_id=os.environ["SLURM_JOB_ID"],
        partition=os.environ["SLURM_JOB_PARTITION"], gpu=gpu_record,
    )
    extension = load_extension(args.extension, args.extension_sha256)
    records = []
    raw = []

    model = _load_model(args.snapshot, device=args.device)
    fp16, samples = _variant_record(
        "fp16", model=model, input_ids=input_ids, prompt=prompt, args=args, torch=torch,
    )
    records.append(fp16)
    raw.extend(samples)
    replaced = replace_all_projection_linears(
        model, extension=extension, rotate_down_projections=False,
        transform_factory=lambda _width: IdentityTransform(), torch_module=torch,
    )
    identity, samples = _variant_record(
        "w4a4_identity", model=model, input_ids=input_ids, prompt=prompt, args=args, torch=torch,
    )
    identity["packed_projection_count"] = len(replaced)
    identity_layer, layer_raw = benchmark_affected_layer(
        model, prompt, transform="I", warmups=args.layer_warmups,
        repetitions=args.layer_repetitions, torch=torch,
    )
    identity["layer"] = identity_layer
    records.append(identity)
    raw.extend(samples)
    raw.extend(layer_raw)
    del model
    _cleanup_cuda(torch)

    model = _load_model(args.snapshot, device=args.device)
    outer, order = load_quarot_outer_matrix(args.quarot_root, 14336, torch)
    outer = outer.to(device=args.device, dtype=torch.float32)

    def transform_factory(width: int):
        if width != 14336:
            raise BenchmarkError(f"full online Hadamard is limited to down-projection width, got {width}")
        return TorchFullHadamard(width, outer, order, torch)

    replaced = replace_all_projection_linears(
        model, extension=extension, rotate_down_projections=True,
        transform_factory=transform_factory, torch_module=torch,
    )
    hfull, samples = _variant_record(
        "w4a4_hfull_down_proj", model=model, input_ids=input_ids,
        prompt=prompt, args=args, torch=torch,
    )
    hfull["packed_projection_count"] = len(replaced)
    hfull_layer, layer_raw = benchmark_affected_layer(
        model, prompt, transform="Hfull", warmups=args.layer_warmups,
        repetitions=args.layer_repetitions, torch=torch,
    )
    hfull["layer"] = hfull_layer
    records.append(hfull)
    raw.extend(samples)
    raw.extend(layer_raw)
    del model
    _cleanup_cuda(torch)

    if tuple(record["variant"] for record in records) != VARIANTS:
        raise BenchmarkError("required comparable model rows are absent or out of order")
    decision = _decision(identity, hfull)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "scientific_evidence": True,
        "recorded_at_utc": _utc(),
        "source": {
            "stage": str(args.stage_root.resolve(strict=True)),
            "commit": args.source_commit,
        },
        "model": {"repository": args.model_repository, "revision": args.model_revision,
                  "snapshot": str(args.snapshot.resolve(strict=True)), "dtype": "float16"},
        "data": {"revision": args.dataset_revision, "arrow": str(args.dataset_arrow.resolve(strict=True)),
                 "arrow_sha256": args.dataset_sha256, "preprocessing": preprocessing},
        "quantization": {
            "weight_bits": 4, "activation_bits": 4,
            "weights": "symmetric per-output-row RTN, max-abs/7",
            "activations": "symmetric dynamic per-token-row RTN, max-abs/7",
            "storage": "signed int4 packed two per uint8",
            "gemm": "CUTLASS int4 x int4 with int32 accumulation",
            "scope": "all 224 Transformer q/k/v/o/gate/up/down projection linears; embeddings/norm/lm_head floating point",
            "rotation_scope": "online full Hadamard and matching folded weight only at all 32 FFN down_proj inputs",
            "fusion": "none",
        },
        "correctness": {"tolerance": args.correctness_tolerance, **smoke},
        "decode_settings": {
            "prompt_length": args.prompt_length, "output_length": args.output_length,
            "batch_size": 1, "warmups": args.decode_warmups,
            "repetitions": args.decode_repetitions,
            "generation": "manual greedy argmax; use_cache=true; exact output length",
        },
        "hardware": {
            "hostname": platform.node(), "job_id": os.environ["SLURM_JOB_ID"],
            "partition": os.environ["SLURM_JOB_PARTITION"], "gpu": gpu_record,
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "transformers": _distribution_version("transformers"),
            "pyarrow": _distribution_version("pyarrow"),
        },
        "variants": records,
        "decision": decision,
    }
    _exclusive_jsonl(args.output_directory / "raw-timings.jsonl", raw)
    result["artifacts"] = {
        "raw_timings": {
            "path": str(args.output_directory / "raw-timings.jsonl"),
            "sha256": sha256_file(args.output_directory / "raw-timings.jsonl"),
            "rows": len(raw),
        }
    }
    _exclusive_json(args.output_directory / "results.json", result)
    _append_event(args.status_path, "scientific_driver_completed", decision=decision["conclusion"])
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--model-repository", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset-arrow", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--extension-sha256", required=True)
    parser.add_argument("--quarot-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--ppl-sequence-length", type=int, default=1024)
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--output-length", type=int, default=32)
    parser.add_argument("--decode-warmups", type=int, default=1)
    parser.add_argument("--decode-repetitions", type=int, default=5)
    parser.add_argument("--layer-warmups", type=int, default=20)
    parser.add_argument("--layer-repetitions", type=int, default=100)
    parser.add_argument("--correctness-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--fusion", choices=("none",), default="none")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output = args.output_directory.resolve()
        if output.exists():
            raise BenchmarkError(f"output directory already exists: {output}")
        output.parent.resolve(strict=True)
        output.mkdir(mode=0o700)
        args.output_directory = output
        _append_event(args.status_path, "scientific_driver_started", output_directory=str(output))
        result = run(args)
    except BaseException as error:
        try:
            _append_event(args.status_path, "runtime_error", error_type=type(error).__name__, error=str(error))
            if "output" in locals() and output.is_dir() and not (output / "failure.json").exists():
                _exclusive_json(output / "failure.json", {
                    "schema_version": "phase-a-real-model-failure-v1",
                    "recorded_at_utc": _utc(), "error_type": type(error).__name__, "error": str(error),
                })
        finally:
            print(f"FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
