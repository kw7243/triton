"""Machine-record schema for unique measured Phase C policies."""

from __future__ import annotations

import math
from typing import Mapping, Sequence


ROW_SCHEMA_VERSION = "phase-c-measured-policy-v1"


class SchemaError(ValueError):
    """A measured policy record is incomplete or incomparable."""


def validate_record(row: Mapping[str, object]) -> None:
    required = {
        "schema_version", "measurement_id", "policy_aliases", "assignment_sha256",
        "assignments", "source_rows", "source_commit", "model", "data", "quantization",
        "quality", "decode", "realized_predicted_transform_cost_ms",
        "realized_predicted_transform_cost_percent", "predicted_aggregate_local_nmse_loss_vs_hfull",
        "fusion", "seed", "hardware",
    }
    if set(row) != required or row.get("schema_version") != ROW_SCHEMA_VERSION:
        raise SchemaError("measured policy top-level schema differs")
    assignments = row["assignments"]
    if set(assignments) != {str(layer) for layer in range(32)}:
        raise SchemaError("policy must assign exactly 32 layers")
    if any(value not in ("I", "H32", "H128", "Hfull") for value in assignments.values()):
        raise SchemaError("policy uses a transform outside the frozen menu")
    if len(row["source_rows"]) != 32 or not row["policy_aliases"]:
        raise SchemaError("policy aliases or source-row hashes are incomplete")
    if row["fusion"] != "none" or row["seed"] != 20260902:
        raise SchemaError("fusion or seed differs")
    quality = row["quality"]
    if quality["dataset"] != "Salesforce/wikitext" or quality["sequence_length"] != 1024:
        raise SchemaError("WikiText-2 PPL settings differ")
    if not math.isfinite(float(quality["perplexity"])) or float(quality["perplexity"]) <= 0:
        raise SchemaError("perplexity must be finite and positive")
    decode = row["decode"]
    if (decode["prompt_length"], decode["output_length"], decode["batch_size"],
            decode["warmups"], decode["repetitions"]) != (128, 32, 1, 1, 5):
        raise SchemaError("accepted decode settings differ")
    for key in ("median_ms", "median_ms_per_output_token", "median_tokens_per_second"):
        if not math.isfinite(float(decode[key])) or float(decode[key]) <= 0:
            raise SchemaError(f"decode {key} must be finite and positive")
    if float(row["realized_predicted_transform_cost_ms"]) < 0:
        raise SchemaError("realized predicted transform cost is negative")


def validate_records(rows: Sequence[Mapping[str, object]], measurement_order: Sequence[str]) -> None:
    for row in rows:
        validate_record(row)
    if [row["measurement_id"] for row in rows] != list(measurement_order):
        raise SchemaError("measured policy order differs from the frozen unique order")
    if len({row["assignment_sha256"] for row in rows}) != len(rows):
        raise SchemaError("deduplicated measurements contain duplicate assignments")
