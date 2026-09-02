#!/usr/bin/env python3
"""One-shot Phase C evaluation of frozen adaptive policies and fixed baselines."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Mapping

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.structured_hadamard.phase_a.real_model import benchmark as phase_a_benchmark
from experiments.structured_hadamard.phase_a.real_model import runtime as phase_a_runtime
from experiments.structured_hadamard.phase_b.runtime import (
    make_transform,
    replace_all_projection_linears_for_policy,
)
from experiments.structured_hadamard.phase_c import analysis, selector
from experiments.structured_hadamard.phase_c.schema import (
    ROW_SCHEMA_VERSION,
    validate_records,
)


SCHEMA_VERSION = "phase-c-selector-result-v1"
SEED = 20260902
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
EVALUATION_SHA256 = "2b8a3efac7b468cbe6432edba5f55c21e435d93873acc6727431f08d5ed328ea"
EXTENSION_SHA256 = "10a961e8855d7349708412aa8c21a38667fb75c94c372180dd6f992913e0a8e0"
PHASE_A_RESULT_SHA256 = "a173ee7fc4d43ab4b33123190ea4e7893c1d5bf421587f07e2502b75fcfd61d4"
PHASE_A_HFULL_PPL = 101.04225822787963


class DriverError(RuntimeError):
    """The single Phase C scientific run is incomplete or incomparable."""


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _append_event(path: Path, event: str, *, announce: bool = False, **values: object) -> None:
    payload = {"recorded_at_utc": _utc(), "event": event, **values}
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
    with os.fdopen(descriptor, "ab") as stream:
        stream.write((json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))
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


def _exclusive_jsonl(path: Path, values: list[object]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        for value in values:
            stream.write((json.dumps(value, sort_keys=True) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def _cleanup_cuda(torch: object) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _verify_phase_a(path: Path) -> None:
    if _sha256(path) != PHASE_A_RESULT_SHA256:
        raise DriverError("accepted Phase A result digest differs")
    value = json.loads(path.read_text(encoding="utf-8"))
    matches = [row for row in value["variants"] if row["variant"] == "w4a4_hfull_down_proj"]
    if len(matches) != 1 or float(matches[0]["quality"]["perplexity"]) != PHASE_A_HFULL_PPL:
        raise DriverError("accepted Phase A Hfull PPL differs")


def _load_freeze(args: argparse.Namespace) -> dict[str, object]:
    if _sha256(args.phase_b_map) != selector.PHASE_B_MAP_SHA256:
        raise DriverError("Phase B map bytes differ")
    if _sha256(args.phase_b_report) != selector.PHASE_B_REPORT_SHA256:
        raise DriverError("Phase B report bytes differ")
    if _sha256(args.policy_freeze) != args.policy_freeze_sha256:
        raise DriverError("policy freeze digest differs")
    value = json.loads(args.policy_freeze.read_text(encoding="utf-8"))
    selector.validate_freeze(value, map_path=args.phase_b_map, report_path=args.phase_b_report)
    if value["mixquant_perq"]["available"] is not False:
        raise DriverError("MixQuant/PeRQ availability changed after policy freeze")
    return value


def _policy_model(args: argparse.Namespace, measurement: Mapping[str, object], *, extension: object,
                  outer: object, outer_order: int, torch: object):
    model = phase_a_runtime.load_cached_llama3(args.snapshot).eval().to(args.device)
    transforms = {
        layer: make_transform(
            measurement["assignments"][str(layer)], 14336, torch_module=torch,
            full_outer_matrix=outer, full_outer_order=outer_order,
        )
        for layer in range(32)
    }
    replaced = replace_all_projection_linears_for_policy(
        model, extension=extension, down_transforms=transforms, torch_module=torch,
    )
    if len(replaced) != 224:
        raise DriverError("policy did not install exactly 224 packed projections")
    return model


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
        raise DriverError(f"accepted extension requires a native SM80/SM86 target, observed {capability}")
    if (args.model_repository, args.model_revision) != (
            phase_a_runtime.MODEL_REPOSITORY, phase_a_runtime.MODEL_REVISION):
        raise DriverError("model identity differs from accepted Phase A/B")
    if args.dataset_revision != DATASET_REVISION or args.evaluation_sha256 != EVALUATION_SHA256:
        raise DriverError("dataset revision or digest argument differs")
    if _sha256(args.evaluation_arrow) != EVALUATION_SHA256:
        raise DriverError("evaluation Arrow bytes differ")
    if args.extension_sha256 != EXTENSION_SHA256 or _sha256(args.extension) != EXTENSION_SHA256:
        raise DriverError("accepted packed-W4A4 runtime differs")
    if args.seed != SEED or args.fusion != "none":
        raise DriverError("seed or sequential fusion contract differs")
    if (args.ppl_sequence_length, args.prompt_length, args.output_length,
            args.decode_warmups, args.decode_repetitions) != (1024, 128, 32, 1, 5):
        raise DriverError("accepted Phase A PPL/decode settings differ")
    _verify_phase_a(args.phase_a_results)
    freeze = _load_freeze(args)
    _append_event(
        args.status_path, "policies_verified_frozen", announce=True,
        policy_freeze_sha256=args.policy_freeze_sha256,
        identities=freeze["deduplication"]["identity_count"],
        unique_measurements=freeze["deduplication"]["unique_measurement_count"],
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    _, evaluation_ids, preprocessing = phase_a_benchmark._tokenize_wikitext(
        args.snapshot, args.evaluation_arrow,
    )
    if evaluation_ids.shape[1] < args.prompt_length:
        raise DriverError("WikiText token stream is shorter than the decode prompt")
    prompt = evaluation_ids[:, :args.prompt_length].to(args.device)

    gpu = torch.cuda.get_device_properties(0)
    hardware = {
        "name": gpu.name,
        "uuid": str(getattr(gpu, "uuid", "unavailable")),
        "total_memory_bytes": int(gpu.total_memory),
        "compute_capability": list(capability),
        "driver": torch._C._cuda_getDriverVersion() if hasattr(torch._C, "_cuda_getDriverVersion") else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "job_id": os.environ["SLURM_JOB_ID"],
        "partition": os.environ["SLURM_JOB_PARTITION"],
        "hostname": platform.node(),
        "direct_phase_a_b_comparison": capability == (8, 6),
    }
    _append_event(args.status_path, "gpu_visible", announce=True, hardware=hardware)
    extension = phase_a_runtime.load_extension(args.extension, args.extension_sha256)
    outer, outer_order = phase_a_runtime.load_quarot_outer_matrix(args.quarot_root, 14336, torch)
    outer = outer.to(device=args.device, dtype=torch.float32)

    measured, raw_timings = [], []
    for measurement_id in freeze["measurement_order"]:
        measurement = freeze["measurements"][measurement_id]
        _append_event(
            args.status_path, "policy_started", measurement_id=measurement_id,
            policy_aliases=measurement["policy_aliases"],
        )
        model = _policy_model(
            args, measurement, extension=extension, outer=outer,
            outer_order=outer_order, torch=torch,
        )
        quality = phase_a_benchmark.evaluate_perplexity(
            model, evaluation_ids, sequence_length=args.ppl_sequence_length,
            device=args.device, torch=torch,
        )
        decode, samples = phase_a_benchmark.benchmark_decode(
            model, prompt, output_length=args.output_length,
            warmups=args.decode_warmups, repetitions=args.decode_repetitions, torch=torch,
        )
        record = {
            "schema_version": ROW_SCHEMA_VERSION,
            "measurement_id": measurement_id,
            "policy_aliases": measurement["policy_aliases"],
            "assignment_sha256": measurement["assignment_sha256"],
            "assignments": measurement["assignments"],
            "source_rows": measurement["source_rows"],
            "source_commit": args.source_commit,
            "model": {"repository": args.model_repository, "revision": args.model_revision,
                      "dtype": "float16"},
            "data": {"dataset": "Salesforce/wikitext", "configuration": "wikitext-2-raw-v1",
                     "revision": args.dataset_revision, "split": "test",
                     "arrow_sha256": args.evaluation_sha256},
            "quantization": {
                "weights": "signed symmetric int4 per output row",
                "activations": "signed symmetric dynamic int4 per token row",
                "accumulation": "int32",
                "runtime": "accepted packed W4A4 CUTLASS extension",
                "runtime_sha256": args.extension_sha256,
                "scope": "all 224 Transformer projection linears; assigned transforms at 32 FFN down_proj inputs only",
            },
            "quality": quality,
            "decode": decode,
            "realized_predicted_transform_cost_ms":
                measurement["realized_predicted_transform_cost_ms"],
            "realized_predicted_transform_cost_percent":
                measurement["realized_predicted_transform_cost_percent"],
            "predicted_aggregate_local_nmse_loss_vs_hfull":
                measurement["predicted_aggregate_local_nmse_loss_vs_hfull"],
            "fusion": args.fusion,
            "seed": args.seed,
            "hardware": hardware,
        }
        measured.append(record)
        raw_timings.extend({
            "measurement_id": measurement_id,
            "policy_aliases": measurement["policy_aliases"],
            "metric": "end_to_end_decode",
            "repetition": index,
            "milliseconds": value,
        } for index, value in enumerate(samples))
        _append_event(
            args.status_path, "policy_completed", measurement_id=measurement_id,
            policy_aliases=measurement["policy_aliases"], perplexity=quality["perplexity"],
            decode_median_ms=decode["median_ms"],
        )
        del model
        _cleanup_cuda(torch)

    validate_records(measured, freeze["measurement_order"])
    decision = analysis.literal_decision(measured, freeze)
    identities = analysis.identity_rows(measured, freeze)
    measured_path = args.output_directory / "measured-policies.jsonl"
    raw_path = args.output_directory / "raw-timings.jsonl"
    assignment_path = args.output_directory / "policy-assignments.json"
    decision_path = args.output_directory / "literal-decision-c.json"
    table_path = args.output_directory / "phase-c-pareto-table.csv"
    figure_path = args.output_directory / "phase-c-five-budget-pareto.svg"
    _exclusive_jsonl(measured_path, measured)
    _exclusive_jsonl(raw_path, raw_timings)
    _exclusive_json(assignment_path, {
        "schema_version": "phase-c-measured-policy-assignments-v1",
        "policy_freeze_sha256": args.policy_freeze_sha256,
        "identity_to_measurement": freeze["identity_to_measurement"],
        "measurement_order": freeze["measurement_order"],
        "measurements": freeze["measurements"],
    })
    _exclusive_json(decision_path, decision)
    analysis.write_table(table_path, identities)
    analysis.write_figure(figure_path, identities)
    artifact_paths = (measured_path, raw_path, assignment_path, decision_path, table_path, figure_path)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "scientific_evidence": True,
        "recorded_at_utc": _utc(),
        "source": {"stage": str(args.stage_root.resolve(strict=True)), "commit": args.source_commit},
        "phase_b": {
            "provenance_tip": selector.PHASE_B_PROVENANCE_TIP,
            "map_sha256": selector.PHASE_B_MAP_SHA256,
            "report_sha256": selector.PHASE_B_REPORT_SHA256,
        },
        "policy_freeze": {
            "path": str(args.policy_freeze.resolve(strict=True)),
            "sha256": args.policy_freeze_sha256,
            "identities": freeze["deduplication"]["identity_count"],
            "unique_measurements": freeze["deduplication"]["unique_measurement_count"],
        },
        "model": {"repository": args.model_repository, "revision": args.model_revision},
        "data": {"revision": args.dataset_revision, "evaluation_sha256": args.evaluation_sha256,
                 "preprocessing": preprocessing},
        "quantization": {
            "weight_bits": 4, "activation_bits": 4,
            "runtime_sha256": args.extension_sha256, "fusion": args.fusion,
            "semantics": "accepted Phase A/B packed signed-W4A4; sequential transform then activation quantization",
        },
        "decode_settings": {
            "prompt_length": args.prompt_length, "output_length": args.output_length,
            "batch_size": 1, "warmups": args.decode_warmups,
            "repetitions": args.decode_repetitions,
            "generation": "manual greedy argmax; use_cache=true; exact output length",
        },
        "hardware": hardware,
        "measurements": measured,
        "policy_identities": identities,
        "mixquant_perq": freeze["mixquant_perq"],
        "literal_decision_C": decision,
        "phase_d_entered": False,
        "second_scientific_run_entered": False,
        "artifact_hashes": {path.name: _sha256(path) for path in artifact_paths},
    }
    _exclusive_json(args.output_directory / "results.json", result)
    _append_event(
        args.status_path, "scientific_driver_completed", announce=True,
        decision=decision["conclusion"],
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--model-repository", default=phase_a_runtime.MODEL_REPOSITORY)
    parser.add_argument("--model-revision", default=phase_a_runtime.MODEL_REVISION)
    parser.add_argument("--evaluation-arrow", type=Path, required=True)
    parser.add_argument("--evaluation-sha256", default=EVALUATION_SHA256)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--extension-sha256", default=EXTENSION_SHA256)
    parser.add_argument("--quarot-root", type=Path, required=True)
    parser.add_argument("--phase-a-results", type=Path, required=True)
    parser.add_argument("--phase-b-map", type=Path, required=True)
    parser.add_argument("--phase-b-report", type=Path, required=True)
    parser.add_argument("--policy-freeze", type=Path, required=True)
    parser.add_argument("--policy-freeze-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--ppl-sequence-length", type=int, default=1024)
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--output-length", type=int, default=32)
    parser.add_argument("--decode-warmups", type=int, default=1)
    parser.add_argument("--decode-repetitions", type=int, default=5)
    parser.add_argument("--fusion", choices=("none",), default="none")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output = args.output_directory.resolve()
        if output.exists():
            raise DriverError(f"output directory already exists: {output}")
        output.parent.resolve(strict=True)
        output.mkdir(mode=0o700)
        args.output_directory = output
        _append_event(args.status_path, "scientific_driver_started", announce=True,
                      output_directory=str(output))
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
