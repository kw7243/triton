"""Machine-row schema validation for the 128 Phase B map tuples."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from .analysis import SUBSET_NAMES
from .reference import TRANSFORMS


ROW_SCHEMA_VERSION = "phase-b-map-row-v1"


class SchemaError(ValueError):
    """A row omits evidence required by the Phase B contract."""


def validate_row(row: Mapping[str, object]) -> None:
    required = {
        "schema_version", "source_commit", "layer", "site", "transform", "model", "data",
        "quantization", "calibration", "shape", "block_semantics", "quality",
        "activation_outlier", "timing", "fusion", "seed", "hardware",
    }
    if set(row) != required or row.get("schema_version") != ROW_SCHEMA_VERSION:
        raise SchemaError("map row top-level schema differs")
    layer = row["layer"]
    if type(layer) is not int or not 0 <= layer < 32:
        raise SchemaError("layer must be in [0, 31]")
    expected_site = f"model.layers.{layer}.mlp.down_proj"
    if row["site"] != expected_site or row["transform"] not in TRANSFORMS:
        raise SchemaError("site or transform differs")
    if row["fusion"] != "none" or row["seed"] != 20260902:
        raise SchemaError("fusion and seed must remain frozen")
    calibration = row["calibration"]
    if calibration["sequences"] != 16 or calibration["tokens_per_sequence"] != 512:
        raise SchemaError("calibration must be exactly 16 by 512")
    if calibration["sampled_token_rows"] != 8192 or calibration["expanded"] is not False:
        raise SchemaError("calibration row cap or expansion flag differs")
    if calibration["dtype"] not in ("float16", "bfloat16"):
        raise SchemaError("cache dtype must be fp16 or bf16")
    if row["shape"]["calibration_input"] != [8192, 14336]:
        raise SchemaError("calibration input shape differs")
    if row["shape"]["timing_input"] != [1, 14336]:
        raise SchemaError("timing input shape differs")
    if row["shape"]["reference_output"] != [8192, 4096]:
        raise SchemaError("reference output shape differs")
    quality = row["quality"]
    if quality["metric"] != "normalized_output_mse" or quality["unit"] != "ratio":
        raise SchemaError("quality metric or unit differs")
    if set(quality["subsets"]) != set(SUBSET_NAMES):
        raise SchemaError("two deterministic disjoint calibration subsets are required")
    for record in (quality["full"], *quality["subsets"].values()):
        if not math.isfinite(float(record["normalized_output_error"])) or record["normalized_output_error"] < 0:
            raise SchemaError("normalized output error must be finite and nonnegative")
    for record in (row["activation_outlier"]["full"], *row["activation_outlier"]["subsets"].values()):
        if not math.isfinite(float(record["max_abs"])) or not math.isfinite(float(record["rms"])):
            raise SchemaError("activation outlier statistics must be finite")
    timing = row["timing"]
    if timing["unit"] != "milliseconds" or timing["warmups"] != 20 or timing["repetitions"] != 100:
        raise SchemaError("timing units, warmups, or repetitions differ")
    if timing["synchronization"] != "CUDA events with terminal torch.cuda.synchronize":
        raise SchemaError("timing synchronization differs")
    if set(timing) != {
            "unit", "warmups", "repetitions", "synchronization", "transform",
            "affected_packed_w4a4_layer"}:
        raise SchemaError("timing fields differ")
    hardware = row["hardware"]
    for key in ("name", "uuid", "compute_capability", "total_memory_bytes", "job_id",
                "partition", "hostname"):
        if key not in hardware:
            raise SchemaError(f"hardware identity lacks {key}")


def validate_rows(rows: Sequence[Mapping[str, object]]) -> None:
    for row in rows:
        validate_row(row)
    tuples = {(row["layer"], row["site"], row["transform"]) for row in rows}
    if len(rows) != 128 or len(tuples) != 128:
        raise SchemaError("map must contain one unique row for each of 32 times four tuples")
