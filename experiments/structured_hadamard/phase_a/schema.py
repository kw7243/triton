"""Fail-closed ``rot-site-v1.phase-a.1`` JSONL record contract."""

from __future__ import annotations

import json
import math
import re
from pathlib import PurePosixPath
from typing import Iterable

from .reference import D_FF, D_MODEL, FULL_K, FULL_Q, TRANSFORM_IDS
from .u172 import MATRIX_DIGEST

SCHEMA_VERSION = "rot-site-v1.phase-a.1"
BASE_COMMIT = "f893845b9b91599ebd3b7a9c7f28164f39c7ed94"
ORACLE_COMMIT = "5008669b08c1f11f9b64d52d16fddd47ca754c5a"

_TOP_LEVEL = {
    "schema_version", "run_id", "code", "model", "quant", "site", "transform", "workload", "hardware",
    "metrics", "timing", "execution", "artifacts", "notes"
}
_CODE = {
    "experiment_commit", "base_commit", "transform_impl_commit", "origin", "intended_upstream", "license",
    "oracle_repo", "oracle_upstream", "oracle_commit", "oracle_license", "dirty"
}
_MODEL = {"id", "revision", "d_model", "d_ff", "n_layers"}
_QUANT = {
    "w_bits", "a_bits", "w_group_size", "a_group_size", "w_symmetric", "a_symmetric", "scale_granularity",
    "clip", "calibration_dataset", "calibration_seed", "calibration_rows"
}
_SITE = {"layer", "kind"}
_TRANSFORM = {
    "id", "axis", "d", "block_size", "K", "q", "normalization", "channel_order", "sign", "permutation",
    "matrix_digest", "fusion", "implementation"
}
_WORKLOAD = {
    "mode", "input_shape", "weight_shape", "batch", "prompt_tokens", "output_tokens", "activation_dtype",
    "contiguous", "input_source", "seed"
}
_HARDWARE = {"gpu", "compute_capability", "driver", "cuda", "torch", "triton", "clock_policy"}
_METRICS = {
    "correctness_passed", "inverse_rel_error", "inverse_max_abs_error", "local_equivalence_rel_error",
    "local_equivalence_max_abs_error", "local_nmse", "nmse_epsilon", "activation_absmax", "activation_rms",
    "ppl_wikitext2", "rotation_us_p10", "rotation_us_median", "rotation_us_p90", "rotation_quantize_us_p10",
    "rotation_quantize_us_median", "rotation_quantize_us_p90", "affected_layer_us_median",
    "end_to_end_ms_per_token", "tokens_per_s"
}
_TIMING = {"identity", "warmup_ms", "repetition_ms", "outer_trials", "timer", "synchronized", "statistics"}
_EXECUTION = {
    "status", "scheduler_clearance", "scientific_evidence", "synthetic_input", "transform_launches",
    "transform_copies", "total_launches"
}
_ARTIFACTS = {"record_jsonl", "raw_samples_jsonl"}
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


class ContractError(ValueError):
    pass


def _object_no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _require_exact_keys(value, expected: set[str], path: str) -> None:
    if not isinstance(value, dict):
        raise ContractError(f"{path} must be an object")
    missing = expected - value.keys()
    unknown = value.keys() - expected
    if missing or unknown:
        raise ContractError(f"{path} keys mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}")


def _string(value, path: str) -> None:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{path} must be a non-empty string")


def _integer(value, path: str, *, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{path} must be an integer >= {minimum}")


def _boolean(value, path: str) -> None:
    if type(value) is not bool:
        raise ContractError(f"{path} must be a boolean")


def _optional_number(value, path: str, *, nonnegative: bool = True) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{path} must be numeric or null")
    if not math.isfinite(value):
        raise ContractError(f"{path} must be finite")
    if nonnegative and value < 0:
        raise ContractError(f"{path} must be nonnegative")


def _reject_nonfinite(value, path: str = "record") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError(f"{path} contains a non-finite number")
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_nonfinite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_nonfinite(child, f"{path}[{index}]")


def _validate_code(code, unexecuted: bool) -> None:
    _require_exact_keys(code, _CODE, "code")
    for key in _CODE - {"dirty"}:
        _string(code[key], f"code.{key}")
    for key in ("experiment_commit", "transform_impl_commit"):
        if not _HEX40.fullmatch(code[key]):
            raise ContractError(f"code.{key} must be a lowercase 40-character commit")
    if code["base_commit"] != BASE_COMMIT:
        raise ContractError("code.base_commit does not match the pinned Triton base")
    if code["oracle_commit"] != ORACLE_COMMIT:
        raise ContractError("code.oracle_commit does not match the frozen QuaRot oracle")
    if code["origin"] != "https://github.com/kw7243/triton.git":
        raise ContractError("code.origin does not match pinned provenance")
    if code["intended_upstream"] != "https://github.com/triton-lang/triton.git":
        raise ContractError("code.intended_upstream does not match pinned provenance")
    if code["license"] != "MIT" or code["oracle_license"] != "Apache-2.0":
        raise ContractError("code licenses do not match pinned provenance")
    _boolean(code["dirty"], "code.dirty")
    if not unexecuted and code["dirty"]:
        raise ContractError("executed measurements must come from a clean commit")


def _validate_transform(transform, timing_identity: str) -> None:
    _require_exact_keys(transform, _TRANSFORM, "transform")
    if transform["id"] not in TRANSFORM_IDS:
        raise ContractError(f"transform.id must be one of {TRANSFORM_IDS}")
    if transform["d"] != D_FF:
        raise ContractError(f"transform.d must be {D_FF}")
    for key, expected in {
        "axis": "last/right", "channel_order": "contiguous-natural", "sign": "none", "permutation": "identity"
    }.items():
        if transform[key] != expected:
            raise ContractError(f"transform.{key} must be {expected!r}")
    expected_fusion = "none" if timing_identity == "transform-only" else "quantize"
    if transform["fusion"] != expected_fusion:
        raise ContractError(f"transform.fusion must be {expected_fusion!r} for {timing_identity}")
    _string(transform["implementation"], "transform.implementation")

    transform_id = transform["id"]
    if transform_id == "I":
        if any(transform[key] is not None for key in ("block_size", "K", "q", "matrix_digest")):
            raise ContractError("identity must not claim block, factorization, or matrix work")
        if transform["normalization"] != "identity" or transform["implementation"] != "host-no-launch":
            raise ContractError("identity must be labeled as host-no-launch with identity normalization")
    elif transform_id in ("H32", "H128"):
        block_size = int(transform_id[1:])
        if transform["block_size"] != block_size or D_FF % block_size:
            raise ContractError(f"{transform_id} requires exact contiguous blocks dividing {D_FF}")
        if any(transform[key] is not None for key in ("K", "q", "matrix_digest")):
            raise ContractError(f"{transform_id} must not claim a full-transform factorization")
        if transform["normalization"] != "orthonormal":
            raise ContractError(f"{transform_id} must be orthonormal")
    else:
        if transform["block_size"] is not None:
            raise ContractError("Hfull must not be labeled as a block transform")
        if transform["K"] != FULL_K or transform["q"] != FULL_Q or transform["K"] * transform["q"] != D_FF:
            raise ContractError("Hfull must use the exact 11008=172x64 factorization")
        if transform["q"] & (transform["q"] - 1):
            raise ContractError("Hfull q must be a power of two")
        if transform["normalization"] != "orthonormal" or transform["matrix_digest"] != MATRIX_DIGEST:
            raise ContractError("Hfull normalization or pinned U_172 digest is inconsistent")


def _validate_execution(record) -> None:
    execution = record["execution"]
    metrics = record["metrics"]
    transform = record["transform"]
    timing_identity = record["timing"]["identity"]
    _require_exact_keys(execution, _EXECUTION, "execution")
    if execution["status"] not in ("unexecuted-plan", "measurement"):
        raise ContractError("execution.status must be unexecuted-plan or measurement")
    for key in ("scheduler_clearance", "scientific_evidence", "synthetic_input"):
        _boolean(execution[key], f"execution.{key}")
    for key in ("transform_launches", "transform_copies", "total_launches"):
        if execution[key] is not None:
            _integer(execution[key], f"execution.{key}")

    unexecuted = execution["status"] == "unexecuted-plan"
    workload_is_synthetic = record["workload"]["input_source"] == "synthetic-fixed-seed"
    if execution["synthetic_input"] is not workload_is_synthetic:
        raise ContractError("execution.synthetic_input must agree with workload.input_source")
    if workload_is_synthetic and execution["scientific_evidence"]:
        raise ContractError("synthetic workloads cannot be labeled scientific evidence")
    if unexecuted:
        if execution["scheduler_clearance"] or execution["scientific_evidence"]:
            raise ContractError("unexecuted plans require scheduler_clearance=false and scientific_evidence=false")
        if any(value is not None for value in metrics.values()):
            raise ContractError("unexecuted plans must use null for every metric")
        if any(execution[key] is not None for key in ("transform_launches", "transform_copies", "total_launches")):
            raise ContractError("unexecuted plans must not present planned launch counts as observations")
        if "UNEXECUTED" not in record["notes"]:
            raise ContractError("unexecuted plans must be visibly labeled in notes")
        return

    if not execution["scheduler_clearance"]:
        raise ContractError("measurements require scheduler clearance")
    if metrics["correctness_passed"] is not True:
        raise ContractError("measurements require a passing correctness gate")
    if any(execution[key] is None for key in ("transform_launches", "transform_copies", "total_launches")):
        raise ContractError("measurements require observed launch/copy counts")
    for key in ("inverse_rel_error", "inverse_max_abs_error", "local_equivalence_rel_error",
                "local_equivalence_max_abs_error", "activation_absmax", "activation_rms"):
        if metrics[key] is None:
            raise ContractError(f"measurements require metrics.{key}")

    if timing_identity == "transform-only":
        required = ("rotation_us_p10", "rotation_us_median", "rotation_us_p90")
        forbidden = ("rotation_quantize_us_p10", "rotation_quantize_us_median", "rotation_quantize_us_p90")
    else:
        required = ("rotation_quantize_us_p10", "rotation_quantize_us_median", "rotation_quantize_us_p90")
        forbidden = ("rotation_us_p10", "rotation_us_median", "rotation_us_p90")
    if any(metrics[key] is None for key in required) or any(metrics[key] is not None for key in forbidden):
        raise ContractError(f"latency metrics are inconsistent with timing identity {timing_identity}")
    if not metrics[required[0]] <= metrics[required[1]] <= metrics[required[2]]:
        raise ContractError("latency quantiles must satisfy p10 <= median <= p90")

    if transform["id"] == "I":
        if execution["transform_launches"] != 0 or execution["transform_copies"] != 0:
            raise ContractError("identity is mislabeled: transform launches and copies must both be zero")
        if timing_identity == "transform-only":
            if execution["total_launches"] != 0 or any(metrics[key] != 0 for key in required):
                raise ContractError("identity transform-only must be an untimed zero-cost host branch")
        elif execution["total_launches"] < 1:
            raise ContractError("identity transform+quantize must observe quantizer work but no transform work")
    elif execution["transform_launches"] != 2 or execution["transform_copies"] != 0:
        raise ContractError("Phase A Hfull must report its two launches and no hidden copies")
    elif execution["total_launches"] < (2 if timing_identity == "transform-only" else 3):
        raise ContractError("Hfull total launches are inconsistent with the timing boundary")


def validate_record(record) -> dict:
    """Validate and return one record, rejecting unknown or inconsistent data."""

    _reject_nonfinite(record)
    _require_exact_keys(record, _TOP_LEVEL, "record")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ContractError(f"schema_version must be {SCHEMA_VERSION!r}")
    _string(record["run_id"], "run_id")
    _string(record["notes"], "notes")

    _require_exact_keys(record["timing"], _TIMING, "timing")
    timing = record["timing"]
    if timing["identity"] not in ("transform-only", "transform+quantize"):
        raise ContractError("timing.identity is invalid")
    for key in ("warmup_ms", "repetition_ms", "outer_trials"):
        _integer(timing[key], f"timing.{key}", minimum=1)
    if timing["timer"] != "device-events" or timing["statistics"] != "p10,p50,p90":
        raise ContractError("timing timer/statistics identity is inconsistent")
    _boolean(timing["synchronized"], "timing.synchronized")

    _require_exact_keys(record["execution"], _EXECUTION, "execution")
    unexecuted = record["execution"]["status"] == "unexecuted-plan"
    _validate_code(record["code"], unexecuted)

    _require_exact_keys(record["model"], _MODEL, "model")
    model = record["model"]
    for key in ("id", "revision"):
        _string(model[key], f"model.{key}")
    if model["id"] != "meta-llama/Llama-2-7b-hf" or model["d_model"] != D_MODEL or model["d_ff"] != D_FF:
        raise ContractError("model identity/dimensions do not match the first-model contract")
    if model["n_layers"] != 32:
        raise ContractError("model.n_layers must be 32")
    if not unexecuted and not _HEX40.fullmatch(model["revision"]):
        raise ContractError("measurements require an immutable 40-character model revision")

    _require_exact_keys(record["quant"], _QUANT, "quant")
    quant = record["quant"]
    if quant["w_bits"] != 4 or quant["a_bits"] != 4:
        raise ContractError("Phase A quant identity must be W4A4")
    for key in ("w_group_size", "a_group_size", "scale_granularity", "clip", "calibration_dataset"):
        _string(quant[key], f"quant.{key}")
    for key in ("w_symmetric", "a_symmetric"):
        _boolean(quant[key], f"quant.{key}")
    _integer(quant["calibration_seed"], "quant.calibration_seed")
    _integer(quant["calibration_rows"], "quant.calibration_rows", minimum=1)
    if not unexecuted:
        for key in ("w_group_size", "a_group_size", "scale_granularity", "clip"):
            if "UNRESOLVED" in quant[key].upper():
                raise ContractError(f"measurements require resolved quant.{key}")
        dataset_name, separator, dataset_revision = quant["calibration_dataset"].rpartition("@")
        if not dataset_name or separator != "@" or not _HEX40.fullmatch(dataset_revision):
            raise ContractError("measurements require calibration_dataset=name@40-character-commit")

    _require_exact_keys(record["site"], _SITE, "site")
    if record["site"]["kind"] != "mlp.down_proj.input":
        raise ContractError("site.kind must be mlp.down_proj.input")
    _integer(record["site"]["layer"], "site.layer")
    if record["site"]["layer"] >= model["n_layers"]:
        raise ContractError("site.layer is outside the model")

    _validate_transform(record["transform"], timing["identity"])

    _require_exact_keys(record["workload"], _WORKLOAD, "workload")
    workload = record["workload"]
    if workload["mode"] != "decode" or workload["input_shape"] != [1, D_FF]:
        raise ContractError("Phase A starts with the exact batch-1 decode shape [1, 11008]")
    if workload["weight_shape"] != [D_MODEL, D_FF]:
        raise ContractError("workload.weight_shape is inconsistent with the stored down_proj weight")
    if workload["batch"] != 1 or workload["prompt_tokens"] != 2048 or workload["output_tokens"] != 128:
        raise ContractError("workload decode identity does not match the frozen plan")
    if workload["activation_dtype"] != "float16" or workload["input_source"] != "synthetic-fixed-seed":
        raise ContractError("workload dtype/input source does not match the Phase A microprofile")
    _boolean(workload["contiguous"], "workload.contiguous")
    if not workload["contiguous"]:
        raise ContractError("Phase A refuses hidden contiguous copies")
    if workload["seed"] != 0:
        raise ContractError("workload.seed must be 0")

    _require_exact_keys(record["hardware"], _HARDWARE, "hardware")
    for key, value in record["hardware"].items():
        _string(value, f"hardware.{key}")
        if unexecuted and value != "UNEXECUTED":
            raise ContractError("unexecuted plans must not invent hardware/software observations")
        if not unexecuted and value == "UNEXECUTED":
            raise ContractError("measurements require observed hardware/software identities")

    _require_exact_keys(record["metrics"], _METRICS, "metrics")
    for key, value in record["metrics"].items():
        if key == "correctness_passed":
            if value is not None:
                _boolean(value, f"metrics.{key}")
        else:
            _optional_number(value, f"metrics.{key}")

    _require_exact_keys(record["artifacts"], _ARTIFACTS, "artifacts")
    for key, value in record["artifacts"].items():
        _string(value, f"artifacts.{key}")
        if not PurePosixPath(value).is_absolute() or not value.endswith(".jsonl"):
            raise ContractError(f"artifacts.{key} must be an absolute JSONL path")

    _validate_execution(record)
    return record


def logical_key(record) -> tuple:
    """Identity for duplicate logical-row rejection within one JSONL artifact."""

    return (
        record["run_id"], record["code"]["experiment_commit"], record["model"]["id"],
        record["model"]["revision"], json.dumps(record["quant"], sort_keys=True, separators=(",", ":")),
        record["site"]["layer"], record["site"]["kind"], record["transform"]["id"],
        record["workload"]["mode"], tuple(record["workload"]["input_shape"]), record["timing"]["identity"]
    )


def validate_records(records: Iterable[dict]) -> list[dict]:
    validated = []
    seen = set()
    for index, record in enumerate(records, start=1):
        validate_record(record)
        key = logical_key(record)
        if key in seen:
            raise ContractError(f"duplicate logical row at record {index}: {key}")
        seen.add(key)
        validated.append(record)
    if not validated:
        raise ContractError("JSONL artifact must contain at least one record")
    return validated


def loads_jsonl(text: str) -> list[dict]:
    records = []
    if not text:
        raise ContractError("JSONL artifact is empty")
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ContractError(f"blank JSONL line {line_number} is not allowed")
        try:
            record = json.loads(line, object_pairs_hook=_object_no_duplicates, parse_constant=lambda value: (_ for _ in ()).throw(
                ContractError(f"non-finite JSON constant {value}")))
        except (json.JSONDecodeError, ContractError) as exc:
            raise ContractError(f"invalid JSONL line {line_number}: {exc}") from exc
        records.append(record)
    return validate_records(records)


def dumps_jsonl(records: Iterable[dict]) -> str:
    validated = validate_records(records)
    return "".join(json.dumps(record, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
                   for record in validated)
