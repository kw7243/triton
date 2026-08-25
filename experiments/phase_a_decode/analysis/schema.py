"""Strict schema and cross-file validation for Phase A decoder results."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "phase-a-analysis-v1"
S_VALUES = (96, 192)
TKV_VALUES = (4096, 16384, 32768)
VARIANTS = ("J", "F", "H")
MODES = ("cold", "hot")
EXPECTED_CELLS = {(s, t) for s in S_VALUES for t in TKV_VALUES}
REQUIRED_FILES = (
    "correctness.json",
    "tuning.json",
    "trial_timings.json",
    "timings.csv",
    "run_metadata.json",
)


class ValidationError(ValueError):
    """An input artifact does not satisfy the Phase A result contract."""


def _fail(where: str, message: str) -> None:
    raise ValidationError(f"{where}: {message}")


def _map(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(where, "expected an object")
    return value


def _array(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(where, "expected an array")
    return value


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(where, "expected a non-empty string")
    return value


def _integer(value: Any, where: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(where, "expected an integer")
    if positive and value <= 0:
        _fail(where, "expected a positive integer")
    return value


def _number(value: Any, where: str, positive: bool = False,
            nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(where, "expected a number")
    result = float(value)
    if not math.isfinite(result):
        _fail(where, "expected a finite number")
    if positive and result <= 0:
        _fail(where, "expected a positive number")
    if nonnegative and result < 0:
        _fail(where, "expected a nonnegative number")
    return result


def _required(obj: dict[str, Any], fields: set[str], where: str) -> None:
    missing = sorted(fields - obj.keys())
    if missing:
        _fail(where, f"missing fields: {', '.join(missing)}")


def _close(actual: float, expected: float, where: str) -> None:
    if not math.isclose(actual, expected, rel_tol=2e-6, abs_tol=1e-9):
        _fail(where, f"{actual!r} does not match derived value {expected!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("JSON", f"duplicate object key {key!r}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(
                handle,
                object_pairs_hook=_unique_object,
                parse_constant=lambda value: _fail(str(path), f"invalid number {value}"),
            )
    except ValidationError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{path}: malformed or unreadable JSON: {exc}") from exc


def validate_correctness(data: Any) -> dict[str, Any]:
    root = _map(data, "correctness.json")
    _required(root, {"status", "bf16", "records"}, "correctness.json")
    if root["status"] != "passed":
        _fail("correctness.json.status", "must be 'passed'")
    bf16 = _string(root["bf16"], "correctness.json.bf16")
    records = _array(root["records"], "correctness.json.records")
    metric_names = {
        f"{variant}_vs_{reference}"
        for variant in ("F", "H")
        for reference in ("J", "oracle", "quantized_oracle")
    }
    seen: set[tuple[str, int, str, str]] = set()
    dtypes: set[str] = set()
    configs: set[str] = set()
    for index, raw in enumerate(records):
        where = f"correctness.json.records[{index}]"
        record = _map(raw, where)
        _required(record, {"dtype", "S", "input", "config", "chunks", "metrics"}, where)
        dtype = _string(record["dtype"], f"{where}.dtype")
        if dtype not in {"float16", "bfloat16"}:
            _fail(f"{where}.dtype", "expected float16 or bfloat16")
        s_value = _integer(record["S"], f"{where}.S", True)
        if s_value not in S_VALUES:
            _fail(f"{where}.S", f"expected one of {S_VALUES}")
        kind = _string(record["input"], f"{where}.input")
        if kind not in {"exhaustive", "random"}:
            _fail(f"{where}.input", "expected exhaustive or random")
        config = _string(record["config"], f"{where}.config")
        _integer(record["chunks"], f"{where}.chunks", True)
        key = (dtype, s_value, kind, config)
        if key in seen:
            _fail(where, f"duplicate correctness case {key}")
        seen.add(key)
        dtypes.add(dtype)
        configs.add(config)
        metrics = _map(record["metrics"], f"{where}.metrics")
        _required(metrics, metric_names, f"{where}.metrics")
        for name in metric_names:
            metric = _map(metrics[name], f"{where}.metrics.{name}")
            _required(metric, {"max_abs", "relative_fro"}, f"{where}.metrics.{name}")
            _number(metric["max_abs"], f"{where}.metrics.{name}.max_abs", nonnegative=True)
            _number(metric["relative_fro"], f"{where}.metrics.{name}.relative_fro",
                    nonnegative=True)
    if "float16" not in dtypes:
        _fail("correctness.json.records", "missing float16 coverage")
    if "unsupported" not in bf16.lower() and "bfloat16" not in dtypes:
        _fail("correctness.json.records", "bf16 reported supported without coverage")
    expected = {
        (dtype, s_value, kind, config)
        for dtype in dtypes
        for s_value in S_VALUES
        for kind in ("exhaustive", "random")
        for config in configs
    }
    if seen != expected:
        _fail(
            "correctness.json.records",
            f"partial/asymmetric matrix; missing={sorted(expected - seen)}, "
            f"extra={sorted(seen - expected)}",
        )
    return {"dtypes": sorted(dtypes), "configs": sorted(configs), "cases": len(records)}


def validate_tuning(data: Any) -> dict[str, Any]:
    root = _map(data, "tuning.json")
    _required(root, {"configs", "selection_shape_tkv", "by_s"}, "tuning.json")
    configs = [
        _string(value, f"tuning.json.configs[{index}]")
        for index, value in enumerate(_array(root["configs"], "tuning.json.configs"))
    ]
    if not configs or len(configs) != len(set(configs)):
        _fail("tuning.json.configs", "expected unique configuration names")
    if _integer(root["selection_shape_tkv"], "tuning.json.selection_shape_tkv", True) != 4096:
        _fail("tuning.json.selection_shape_tkv", "expected 4096")
    by_s = _map(root["by_s"], "tuning.json.by_s")
    if set(by_s) != {str(value) for value in S_VALUES}:
        _fail("tuning.json.by_s", f"expected exactly {S_VALUES}")
    selections: dict[int, dict[str, str]] = {}
    for s_value in S_VALUES:
        where = f"tuning.json.by_s.{s_value}"
        entry = _map(by_s[str(s_value)], where)
        _required(entry, {"scores_ms", "selected"}, where)
        scores = _map(entry["scores_ms"], f"{where}.scores_ms")
        selected = _map(entry["selected"], f"{where}.selected")
        if set(scores) != set(VARIANTS) or set(selected) != set(VARIANTS):
            _fail(where, f"expected exactly variants {VARIANTS}")
        selections[s_value] = {}
        for variant in VARIANTS:
            variant_scores = _map(scores[variant], f"{where}.scores_ms.{variant}")
            if set(variant_scores) != set(configs):
                _fail(f"{where}.scores_ms.{variant}", "configuration set mismatch")
            numeric = {
                name: _number(value, f"{where}.scores_ms.{variant}.{name}", True)
                for name, value in variant_scores.items()
            }
            choice = _string(selected[variant], f"{where}.selected.{variant}")
            if choice not in numeric:
                _fail(f"{where}.selected.{variant}", "unknown configuration")
            if numeric[choice] != min(numeric.values()):
                _fail(f"{where}.selected.{variant}", "selection is not a minimum score")
            selections[s_value][variant] = choice
    return {"configs": configs, "selected": selections}


def _triplet(value: Any, where: str) -> tuple[float, float, float]:
    values = _array(value, where)
    if len(values) != 3:
        _fail(where, "expected [p20, p50, p80]")
    result = tuple(_number(item, f"{where}[{index}]", True)
                   for index, item in enumerate(values))
    if not result[0] <= result[1] <= result[2]:
        _fail(where, "quantiles must be monotonic")
    return result


def validate_trials(data: Any) -> dict[tuple[int, int], list[dict[str, Any]]]:
    root = _map(data, "trial_timings.json")
    parsed: dict[tuple[int, int], list[dict[str, Any]]] = {}
    pattern = re.compile(r"S([0-9]+)-T([0-9]+)$")
    for key, raw_attempts in root.items():
        match = pattern.fullmatch(key)
        if match is None:
            _fail("trial_timings.json", f"unexpected key {key!r}")
        cell = (int(match.group(1)), int(match.group(2)))
        attempts = _array(raw_attempts, f"trial_timings.json.{key}")
        if not attempts:
            _fail(f"trial_timings.json.{key}", "no attempts")
        normalized: list[dict[str, Any]] = []
        previous_rep = 0
        for attempt_index, raw_attempt in enumerate(attempts):
            where = f"trial_timings.json.{key}[{attempt_index}]"
            attempt = _map(raw_attempt, where)
            _required(attempt, {"rep_ms", "trials", "aggregate", "hot_stability"}, where)
            rep_ms = _integer(attempt["rep_ms"], f"{where}.rep_ms", True)
            if rep_ms < previous_rep:
                _fail(f"{where}.rep_ms", "must be nondecreasing")
            previous_rep = rep_ms
            trials = _array(attempt["trials"], f"{where}.trials")
            if len(trials) != 5:
                _fail(f"{where}.trials", "expected five outer trials")
            trial_values = []
            for trial_index, raw_trial in enumerate(trials):
                trial_where = f"{where}.trials[{trial_index}]"
                trial = _map(raw_trial, trial_where)
                _required(trial, set(VARIANTS) | {"order"}, trial_where)
                order = _array(trial["order"], f"{trial_where}.order")
                if len(order) != 3 or set(order) != set(VARIANTS):
                    _fail(f"{trial_where}.order", f"expected a permutation of {VARIANTS}")
                normalized_trial = {}
                for variant in VARIANTS:
                    variant_data = _map(trial[variant], f"{trial_where}.{variant}")
                    _required(variant_data, set(MODES), f"{trial_where}.{variant}")
                    normalized_trial[variant] = {
                        mode: _triplet(variant_data[mode], f"{trial_where}.{variant}.{mode}")
                        for mode in MODES
                    }
                trial_values.append(normalized_trial)
            aggregate = _map(attempt["aggregate"], f"{where}.aggregate")
            if set(aggregate) != set(VARIANTS):
                _fail(f"{where}.aggregate", f"expected {VARIANTS}")
            aggregate_values = {}
            for variant in VARIANTS:
                variant_aggregate = _map(aggregate[variant], f"{where}.aggregate.{variant}")
                _required(variant_aggregate, set(MODES), f"{where}.aggregate.{variant}")
                aggregate_values[variant] = {}
                for mode in MODES:
                    observed = _triplet(
                        variant_aggregate[mode], f"{where}.aggregate.{variant}.{mode}"
                    )
                    expected = tuple(
                        statistics.median(trial[variant][mode][index] for trial in trial_values)
                        for index in range(3)
                    )
                    for index, (actual, derived) in enumerate(zip(observed, expected)):
                        _close(actual, derived,
                               f"{where}.aggregate.{variant}.{mode}[{index}]")
                    aggregate_values[variant][mode] = observed
            stability = _map(attempt["hot_stability"], f"{where}.hot_stability")
            _required(stability, {"J", "H"}, f"{where}.hot_stability")
            for variant in ("J", "H"):
                observed = _number(
                    stability[variant], f"{where}.hot_stability.{variant}", nonnegative=True
                )
                quantiles = aggregate_values[variant]["hot"]
                _close(observed, (quantiles[2] - quantiles[0]) / quantiles[1],
                       f"{where}.hot_stability.{variant}")
            normalized.append(
                {"rep_ms": rep_ms, "aggregate": aggregate_values, "trials": len(trials)}
            )
        parsed[cell] = normalized
    if set(parsed) != EXPECTED_CELLS:
        _fail(
            "trial_timings.json",
            f"partial matrix; missing={sorted(EXPECTED_CELLS - set(parsed))}, "
            f"extra={sorted(set(parsed) - EXPECTED_CELLS)}",
        )
    return parsed


def _csv_float(row: dict[str, str], field: str, line: int,
               positive: bool = False, nonnegative: bool = False) -> float:
    raw = row.get(field)
    if raw is None or raw == "":
        _fail(f"timings.csv:{line}.{field}", "missing value")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValidationError(f"timings.csv:{line}.{field}: expected a number") from exc
    return _number(value, f"timings.csv:{line}.{field}", positive, nonnegative)


def _csv_int(row: dict[str, str], field: str, line: int, positive: bool = False) -> int:
    raw = row.get(field)
    if raw is None or re.fullmatch(r"-?[0-9]+", raw) is None:
        _fail(f"timings.csv:{line}.{field}", "expected an integer")
    return _integer(int(raw), f"timings.csv:{line}.{field}", positive)


def validate_timings(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, restkey="__extra__")
            if reader.fieldnames is None:
                _fail("timings.csv", "missing header")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                _fail("timings.csv", "duplicate header")
            raw_rows = list(reader)
    except OSError as exc:
        raise ValidationError(f"timings.csv: unreadable: {exc}") from exc
    base = {
        "S", "Tkv", "roles", "kv_heads", "head_dim", "chunks", "output_bytes",
        "warmup_ms", "rep_ms", "outer_trials", "j_config", "f_config", "h_config",
        "jh_hot_speedup", "jh_cold_speedup", "stable",
    }
    timing = {
        f"{variant.lower()}_{mode}_{suffix}"
        for variant in VARIANTS
        for mode in MODES
        for suffix in (
            "p20_ms", "p50_ms", "p80_ms", "gchunks_s", "output_gib_s", "stability"
        )
    }
    correctness = {
        f"{variant.lower()}_{metric}_vs_{reference}"
        for variant in ("F", "H")
        for metric in ("max_abs", "relative_fro")
        for reference in ("j", "oracle")
    }
    required = base | timing | correctness | {"h_max_abs", "h_relative_fro"}
    missing = sorted(required - set(reader.fieldnames))
    if missing:
        _fail("timings.csv", f"missing columns: {', '.join(missing)}")
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    for line, raw in enumerate(raw_rows, 2):
        if raw.get("__extra__"):
            _fail(f"timings.csv:{line}", "too many values")
        s_value = _csv_int(raw, "S", line, True)
        tkv = _csv_int(raw, "Tkv", line, True)
        cell = (s_value, tkv)
        if cell in rows:
            _fail(f"timings.csv:{line}", f"duplicate cell {cell}")
        row: dict[str, Any] = {"S": s_value, "Tkv": tkv}
        for field in (
            "roles", "kv_heads", "head_dim", "chunks", "output_bytes",
            "warmup_ms", "rep_ms", "outer_trials",
        ):
            row[field] = _csv_int(raw, field, line, True)
        if (
            row["roles"], row["kv_heads"], row["head_dim"],
            row["warmup_ms"], row["outer_trials"],
        ) != (2, 8, 128, 25, 5):
            _fail(f"timings.csv:{line}", "does not match Phase A protocol")
        chunks = 2 * 8 * tkv * 128 // 4
        if row["chunks"] != chunks or row["output_bytes"] != chunks * 8:
            _fail(f"timings.csv:{line}", "chunks/output_bytes do not match shape")
        for variant in VARIANTS:
            config_field = f"{variant.lower()}_config"
            row[config_field] = _string(raw[config_field], f"timings.csv:{line}.{config_field}")
            for mode in MODES:
                prefix = f"{variant.lower()}_{mode}"
                quantiles = tuple(
                    _csv_float(raw, f"{prefix}_{name}_ms", line, True)
                    for name in ("p20", "p50", "p80")
                )
                if not quantiles[0] <= quantiles[1] <= quantiles[2]:
                    _fail(f"timings.csv:{line}.{prefix}", "quantiles are not monotonic")
                row[f"{prefix}_quantiles_ms"] = quantiles
                stability = _csv_float(raw, f"{prefix}_stability", line, nonnegative=True)
                _close(stability, (quantiles[2] - quantiles[0]) / quantiles[1],
                       f"timings.csv:{line}.{prefix}_stability")
                row[f"{prefix}_stability"] = stability
                seconds = quantiles[1] / 1000
                _close(
                    _csv_float(raw, f"{prefix}_gchunks_s", line, True),
                    chunks / seconds / 1e9,
                    f"timings.csv:{line}.{prefix}_gchunks_s",
                )
                _close(
                    _csv_float(raw, f"{prefix}_output_gib_s", line, True),
                    row["output_bytes"] / seconds / 2**30,
                    f"timings.csv:{line}.{prefix}_output_gib_s",
                )
        for field in correctness | {"h_max_abs", "h_relative_fro"}:
            row[field] = _csv_float(raw, field, line, nonnegative=True)
        _close(row["h_max_abs"], max(row["h_max_abs_vs_j"], row["h_max_abs_vs_oracle"]),
               f"timings.csv:{line}.h_max_abs")
        _close(
            row["h_relative_fro"],
            max(row["h_relative_fro_vs_j"], row["h_relative_fro_vs_oracle"]),
            f"timings.csv:{line}.h_relative_fro",
        )
        for mode in MODES:
            ratio = row[f"j_{mode}_quantiles_ms"][1] / row[f"h_{mode}_quantiles_ms"][1]
            row[f"jh_{mode}_speedup"] = _csv_float(
                raw, f"jh_{mode}_speedup", line, True
            )
            _close(row[f"jh_{mode}_speedup"], ratio,
                   f"timings.csv:{line}.jh_{mode}_speedup")
        stable_text = raw["stable"].strip().lower()
        if stable_text not in {"true", "false"}:
            _fail(f"timings.csv:{line}.stable", "expected True or False")
        row["stable"] = stable_text == "true"
        derived_stable = max(row["j_hot_stability"], row["h_hot_stability"]) <= 0.05
        if row["stable"] != derived_stable:
            _fail(f"timings.csv:{line}.stable", "does not match J/H hot stability")
        rows[cell] = row
    if set(rows) != EXPECTED_CELLS:
        _fail(
            "timings.csv",
            f"partial matrix; missing={sorted(EXPECTED_CELLS - set(rows))}, "
            f"extra={sorted(set(rows) - EXPECTED_CELLS)}",
        )
    return rows


def validate_provenance(data: Any) -> dict[str, Any]:
    root = _map(data, "run_metadata.json")
    fields = {
        "status", "git_commit", "git_branch", "cwd", "torch", "triton",
        "cuda_runtime", "device", "compute_capability", "assumptions",
    }
    _required(root, fields, "run_metadata.json")
    if root["status"] != "passed":
        _fail("run_metadata.json.status", "must be 'passed'")
    commit = _string(root["git_commit"], "run_metadata.json.git_commit")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        _fail("run_metadata.json.git_commit", "expected full lowercase 40-hex commit")
    branch = _string(root["git_branch"], "run_metadata.json.git_branch")
    cwd = _string(root["cwd"], "run_metadata.json.cwd")
    if not Path(cwd).is_absolute():
        _fail("run_metadata.json.cwd", "expected an absolute path")
    for field in ("torch", "triton", "cuda_runtime", "device"):
        _string(root[field], f"run_metadata.json.{field}")
    capability = _array(root["compute_capability"], "run_metadata.json.compute_capability")
    if len(capability) != 2:
        _fail("run_metadata.json.compute_capability", "expected [major, minor]")
    _integer(capability[0], "run_metadata.json.compute_capability[0]", True)
    _integer(capability[1], "run_metadata.json.compute_capability[1]")
    assumptions = _map(root["assumptions"], "run_metadata.json.assumptions")
    if (
        assumptions.get("quaternion") != "scalar-first (w,x,y,z)"
        or assumptions.get("id") != "p*S+s"
    ):
        _fail("run_metadata.json.assumptions", "conventions do not match Phase A")
    synthetic = root.get("synthetic", False)
    if not isinstance(synthetic, bool):
        _fail("run_metadata.json.synthetic", "expected a boolean")
    fixture_seed = None
    if synthetic:
        fixture_seed = _integer(root.get("fixture_seed"), "run_metadata.json.fixture_seed")
        if root.get("schema_version") != SCHEMA_VERSION:
            _fail("run_metadata.json.schema_version", f"expected {SCHEMA_VERSION}")
    return {
        "git_commit": commit,
        "git_branch": branch,
        "cwd": cwd,
        "synthetic": synthetic,
        "fixture_seed": fixture_seed,
    }


def validate_result_directory(path: Path) -> dict[str, Any]:
    """Validate required files and their cross-file invariants."""
    path = Path(path)
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        _fail(str(path), f"missing required artifacts: {', '.join(missing)}")
    correctness = validate_correctness(load_json(path / "correctness.json"))
    tuning = validate_tuning(load_json(path / "tuning.json"))
    trials = validate_trials(load_json(path / "trial_timings.json"))
    timings = validate_timings(path / "timings.csv")
    provenance = validate_provenance(load_json(path / "run_metadata.json"))
    if set(correctness["configs"]) != set(tuning["configs"]):
        _fail("cross-file", "correctness and tuning configurations differ")
    for cell, row in timings.items():
        s_value, _ = cell
        final = trials[cell][-1]
        if row["rep_ms"] != final["rep_ms"]:
            _fail(f"cross-file {cell}", "rep_ms differs from final attempt")
        for variant in VARIANTS:
            field = f"{variant.lower()}_config"
            if row[field] != tuning["selected"][s_value][variant]:
                _fail(f"cross-file {cell}", f"{field} differs from tuning")
            for mode in MODES:
                observed = row[f"{variant.lower()}_{mode}_quantiles_ms"]
                expected = final["aggregate"][variant][mode]
                for index, (actual, derived) in enumerate(zip(observed, expected)):
                    _close(actual, derived,
                           f"cross-file {cell}.{variant}.{mode}[{index}]")
    return {
        "schema_version": SCHEMA_VERSION,
        "path": path,
        "correctness": correctness,
        "tuning": tuning,
        "trials": trials,
        "timings": timings,
        "provenance": provenance,
    }
