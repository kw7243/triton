"""Owner-cleared, stage-only Phase A synthetic CUDA microprofile driver."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Callable, Iterable

from .activation_quantizer import W4A4ActivationQuantizer, quantize_rows_reference
from .oracle import run_oracle
from .preflight import RUN_ID, build_plan
from .profiler import ProfileSummary, profile_transform
from .reference import D_FF, forward_rows
from .schema import dumps_jsonl, loads_jsonl, validate_records
from .stage_repository import (
    DEFAULT_EXCLUDED_DIRECTORY_NAMES,
    DEFAULT_EXCLUDED_ROOT_DIRECTORIES,
    STAGE_MANIFEST_NAME,
    STAGE_MANIFEST_SCHEMA_VERSION,
    STAGE_METADATA_NAME,
    STAGE_SCHEMA_VERSION,
    _manifest,
    _source_paths,
)
from .triton_transform import HFullWorkspace, apply_transform

INTAKE_COMMIT = "d57acb60db2a4507bbff984fb3c9771e8a6ada3d"
CLEARANCE_SCHEMA_VERSION = "phase-a-scheduler-clearance-v1"
RAW_SCHEMA_VERSION = "phase-a-raw-timing-v1"
OUTPUT_SCHEMA_VERSION = "phase-a-execution-output-v1"
RECORD_NAME = "phase-a.jsonl"
RAW_NAME = "phase-a.raw-samples.jsonl"
OUTPUT_MANIFEST_NAME = "phase-a.execution.json"

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_CUDA_DEVICE = re.compile(r"^cuda:[0-9]+$")
_CLEARANCE_KEYS = {
    "schema_version", "scheduler_clearance", "clearance_id", "owner", "owner_uid", "stage_root",
    "source_commit", "driver_commit", "transform_commit", "stage_manifest_sha256", "device",
    "output_directory", "model_revision", "site_layer", "quant", "timing", "clock_policy",
}
_QUANT_KEYS = {
    "w_bits", "a_bits", "w_group_size", "a_group_size", "w_symmetric", "a_symmetric",
    "scale_granularity", "clip", "calibration_dataset", "calibration_seed", "calibration_rows",
}
_TIMING_KEYS = {"warmup_ms", "repetition_ms", "outer_trials"}
_STAGE_METADATA_KEYS = {
    "schema_version", "source_head", "source_git_common_dir", "staged_git_common_dir",
    "tracked_and_untracked_entries", "default_exclusions", "working_tree_manifest",
    "working_tree_manifest_sha256", "working_tree_manifest_file_sha256", "command",
}


class ExecutionRefusal(RuntimeError):
    """A provenance, authorization, configuration, or output gate failed."""


@dataclass(frozen=True)
class StageIdentity:
    root: Path
    head: str
    manifest_sha256: str
    metadata_sha256: str


@dataclass(frozen=True)
class Authorization:
    clearance_id: str
    owner: str
    clearance_sha256: str
    config_sha256: str
    stage: StageIdentity
    device: str
    output_directory: Path
    model_revision: str
    site_layer: int
    quant: dict
    timing: dict
    clock_policy: str


def _run_git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(("git", *args), cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, check=False)
    if check and completed.returncode:
        raise ExecutionRefusal(f"Git verification failed: {' '.join(args)}: {completed.stderr.strip()}")
    return completed


def _absolute_git_path(root: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _object_no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ExecutionRefusal(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json_bytes(value: bytes, label: str) -> dict:
    try:
        loaded = json.loads(value.decode("utf-8"), object_pairs_hook=_object_no_duplicates,
                            parse_constant=lambda token: (_ for _ in ()).throw(
                                ExecutionRefusal(f"non-finite JSON constant {token}")))
    except (UnicodeDecodeError, json.JSONDecodeError, ExecutionRefusal) as exc:
        raise ExecutionRefusal(f"invalid {label}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ExecutionRefusal(f"{label} must be a JSON object")
    return loaded


def _exact_keys(value: dict, expected: set[str], label: str) -> None:
    missing = expected - value.keys()
    unknown = value.keys() - expected
    if missing or unknown:
        raise ExecutionRefusal(f"{label} keys mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}")


def _regular_file_bytes(path: Path, label: str) -> bytes:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise ExecutionRefusal(f"required {label} is absent: {path}") from exc
    if not stat.S_ISREG(mode) or path.is_symlink():
        raise ExecutionRefusal(f"{label} must be a regular non-symlink file: {path}")
    return path.read_bytes()


def _inside(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath((path, root)) == str(root)
    except ValueError:
        return False


def verify_complete_stage(root: str | os.PathLike[str]) -> StageIdentity:
    """Prove an ordinary, complete, clean stage without importing CUDA dependencies."""

    stage_root = Path(root).resolve(strict=True)
    top = Path(_run_git(stage_root, "rev-parse", "--show-toplevel").stdout.strip()).resolve()
    if top != stage_root:
        raise ExecutionRefusal(f"execution must start at the exact staged repository root: {top}")
    if not (stage_root / ".git").is_dir():
        raise ExecutionRefusal("stage .git must be an ordinary directory")
    git_dir = _absolute_git_path(stage_root, _run_git(stage_root, "rev-parse", "--git-dir").stdout.strip())
    common_dir = _absolute_git_path(stage_root,
                                    _run_git(stage_root, "rev-parse", "--git-common-dir").stdout.strip())
    if git_dir != stage_root / ".git" or common_dir != stage_root / ".git":
        raise ExecutionRefusal("stage Git metadata is not self-contained")
    if _run_git(stage_root, "rev-parse", "--is-shallow-repository").stdout.strip() != "false":
        raise ExecutionRefusal("shallow stages are not complete repositories")
    promisor = _run_git(stage_root, "config", "--get-regexp", r"^remote\..*\.promisor$", check=False)
    if promisor.returncode == 0 and promisor.stdout.strip():
        raise ExecutionRefusal("partial/promisor stages are not complete repositories")
    alternates = stage_root / ".git" / "objects" / "info" / "alternates"
    if alternates.exists() and alternates.read_text(encoding="utf-8").strip():
        raise ExecutionRefusal("stage object database uses an external alternate")

    head = _run_git(stage_root, "rev-parse", "--verify", "HEAD^{commit}").stdout.strip()
    if not _HEX40.fullmatch(head):
        raise ExecutionRefusal("stage HEAD is not a full commit id")
    _run_git(stage_root, "fsck", "--connectivity-only", "--no-dangling")

    metadata_bytes = _regular_file_bytes(stage_root / STAGE_METADATA_NAME, "stage metadata")
    manifest_bytes = _regular_file_bytes(stage_root / STAGE_MANIFEST_NAME, "stage manifest")
    metadata = _load_json_bytes(metadata_bytes, "stage metadata")
    manifest = _load_json_bytes(manifest_bytes, "stage manifest")
    _exact_keys(metadata, _STAGE_METADATA_KEYS, "stage metadata")
    _exact_keys(manifest, {"schema_version", "entries"}, "stage manifest")
    if metadata["schema_version"] != STAGE_SCHEMA_VERSION:
        raise ExecutionRefusal(f"stage metadata schema must be {STAGE_SCHEMA_VERSION}")
    if manifest["schema_version"] != STAGE_MANIFEST_SCHEMA_VERSION or not isinstance(manifest["entries"], dict):
        raise ExecutionRefusal("stage manifest schema or entries are invalid")
    entries = manifest["entries"]
    if metadata["source_head"] != head:
        raise ExecutionRefusal("stage metadata source_head does not equal executed HEAD")
    if metadata["staged_git_common_dir"] != str(stage_root / ".git"):
        raise ExecutionRefusal("stage metadata points at different Git metadata")
    if Path(metadata["source_git_common_dir"]).resolve() == common_dir:
        raise ExecutionRefusal("stage metadata does not prove independent Git metadata")
    if metadata["working_tree_manifest"] != STAGE_MANIFEST_NAME:
        raise ExecutionRefusal("stage metadata names an unexpected manifest")
    if metadata["working_tree_manifest_file_sha256"] != _bytes_sha256(manifest_bytes):
        raise ExecutionRefusal("stage manifest file digest mismatch")
    if metadata["working_tree_manifest_sha256"] != _canonical_sha256(entries):
        raise ExecutionRefusal("stage working-tree manifest digest mismatch")
    if metadata["tracked_and_untracked_entries"] != len(entries):
        raise ExecutionRefusal("stage manifest entry count mismatch")
    expected_exclusions = {
        "root_directories": list(DEFAULT_EXCLUDED_ROOT_DIRECTORIES),
        "directory_names_at_any_depth": list(DEFAULT_EXCLUDED_DIRECTORY_NAMES),
    }
    if metadata["default_exclusions"] != expected_exclusions:
        raise ExecutionRefusal("stage exclusion policy does not match the executing driver")
    if not isinstance(metadata["command"], list) or not all(isinstance(item, str) for item in metadata["command"]):
        raise ExecutionRefusal("stage metadata command must be a string list")

    tracked = set(filter(None, _run_git(stage_root, "ls-files", "-z").stdout.split("\0")))
    if set(entries) != tracked:
        raise ExecutionRefusal("executed stage manifest must contain exactly the committed tracked paths")
    if any(not isinstance(identity, dict) or identity.get("kind") not in ("file", "symlink")
           for identity in entries.values()):
        raise ExecutionRefusal("executed stage manifest contains missing or invalid entries")
    paths = tuple(path for path in _source_paths(stage_root)
                  if path.as_posix() not in (STAGE_METADATA_NAME, STAGE_MANIFEST_NAME))
    if _manifest(stage_root, paths) != entries:
        raise ExecutionRefusal("stage content changed after its manifest was recorded")
    dirty = _run_git(stage_root, "status", "--porcelain=v1", "--untracked-files=all").stdout.splitlines()
    allowed = {f"?? {STAGE_METADATA_NAME}", f"?? {STAGE_MANIFEST_NAME}"}
    if any(line not in allowed for line in dirty):
        raise ExecutionRefusal("executed commit is dirty or has mutable untracked source")
    return StageIdentity(stage_root, head, metadata["working_tree_manifest_sha256"],
                         _bytes_sha256(metadata_bytes))


def _validate_quant(quant: object) -> dict:
    if not isinstance(quant, dict):
        raise ExecutionRefusal("clearance quant must be an object")
    _exact_keys(quant, _QUANT_KEYS, "clearance quant")
    for key in ("w_bits", "a_bits", "calibration_seed", "calibration_rows"):
        if type(quant[key]) is not int:
            raise ExecutionRefusal(f"clearance quant.{key} must be an integer")
    if quant["w_bits"] != 4 or quant["a_bits"] != 4:
        raise ExecutionRefusal("clearance quantization must be W4A4")
    if quant["a_group_size"] != "per-row" or quant["a_symmetric"] is not True:
        raise ExecutionRefusal("clearance must select dynamic per-row symmetric A4")
    if type(quant["w_symmetric"]) is not bool:
        raise ExecutionRefusal("clearance quant.w_symmetric must be boolean")
    for key in ("w_group_size", "scale_granularity", "clip", "calibration_dataset"):
        if not isinstance(quant[key], str) or not quant[key] or "UNRESOLVED" in quant[key].upper():
            raise ExecutionRefusal(f"clearance quant.{key} must be resolved")
    dataset, separator, revision = quant["calibration_dataset"].rpartition("@")
    if not dataset or separator != "@" or not _HEX40.fullmatch(revision):
        raise ExecutionRefusal("calibration_dataset must be name@40-character-commit")
    if quant["calibration_seed"] != 0:
        raise ExecutionRefusal("calibration_seed must remain 0")
    if quant["calibration_rows"] <= 0:
        raise ExecutionRefusal("calibration_rows must be a positive integer")
    return copy.deepcopy(quant)


def _validate_timing(timing: object) -> dict:
    if not isinstance(timing, dict):
        raise ExecutionRefusal("clearance timing must be an object")
    _exact_keys(timing, _TIMING_KEYS, "clearance timing")
    expected = {"warmup_ms": 25, "repetition_ms": 200, "outer_trials": 5}
    for key in expected:
        if type(timing[key]) is not int:
            raise ExecutionRefusal(f"clearance timing.{key} must be an integer")
    if timing != expected:
        raise ExecutionRefusal(f"clearance timing must equal the accepted fixed config {expected}")
    return copy.deepcopy(timing)


def load_authorization(path: str | os.PathLike[str], stage: StageIdentity) -> Authorization:
    clearance_path = Path(path)
    if not clearance_path.is_absolute():
        raise ExecutionRefusal("scheduler clearance file path must be absolute")
    try:
        submitted_status = clearance_path.lstat()
    except OSError as error:
        raise ExecutionRefusal("scheduler clearance file is not accessible") from error
    if not stat.S_ISREG(submitted_status.st_mode):
        raise ExecutionRefusal("scheduler clearance must be a regular non-symlink file")
    try:
        descriptor = os.open(clearance_path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise ExecutionRefusal("scheduler clearance must be a regular non-symlink file") from error
    with os.fdopen(descriptor, "rb") as clearance_file:
        opened_status = os.fstat(clearance_file.fileno())
        submitted_identity = (submitted_status.st_dev, submitted_status.st_ino)
        opened_identity = (opened_status.st_dev, opened_status.st_ino)
        if not stat.S_ISREG(opened_status.st_mode) or opened_identity != submitted_identity:
            raise ExecutionRefusal("scheduler clearance changed while being opened")
        try:
            resolved_path = clearance_path.resolve(strict=True)
            resolved_status = resolved_path.stat()
        except (OSError, RuntimeError) as error:
            raise ExecutionRefusal("scheduler clearance changed while being opened") from error
        if (resolved_status.st_dev, resolved_status.st_ino) != opened_identity:
            raise ExecutionRefusal("scheduler clearance changed while being opened")
        if _inside(resolved_path, stage.root):
            raise ExecutionRefusal("scheduler clearance must be outside the immutable stage")
        if opened_status.st_uid != os.geteuid() or stat.S_IMODE(opened_status.st_mode) != 0o600:
            raise ExecutionRefusal("scheduler clearance must be owned by the executing uid with mode 0600")
        clearance_bytes = clearance_file.read()
    value = _load_json_bytes(clearance_bytes, "scheduler clearance")
    _exact_keys(value, _CLEARANCE_KEYS, "scheduler clearance")
    if value["schema_version"] != CLEARANCE_SCHEMA_VERSION or value["scheduler_clearance"] is not True:
        raise ExecutionRefusal("explicit scheduler_clearance=true is required")
    for key in ("clearance_id", "owner", "clock_policy"):
        if not isinstance(value[key], str) or not value[key]:
            raise ExecutionRefusal(f"scheduler clearance {key} must be a non-empty string")
    if isinstance(value["owner_uid"], bool) or value["owner_uid"] != os.geteuid():
        raise ExecutionRefusal("scheduler clearance owner_uid does not match the executing uid")
    if not isinstance(value["stage_root"], str) or Path(value["stage_root"]).resolve() != stage.root:
        raise ExecutionRefusal("scheduler clearance stage_root mismatch")
    for key in ("source_commit", "driver_commit", "transform_commit"):
        if value[key] != stage.head:
            raise ExecutionRefusal(f"scheduler clearance {key} does not equal executed HEAD")
    if not isinstance(value["stage_manifest_sha256"], str) \
            or value["stage_manifest_sha256"] != stage.manifest_sha256:
        raise ExecutionRefusal("scheduler clearance stage manifest mismatch")
    if not isinstance(value["device"], str) or not _CUDA_DEVICE.fullmatch(value["device"]):
        raise ExecutionRefusal("device must select one explicit CUDA index such as cuda:0")
    if not isinstance(value["model_revision"], str) or not _HEX40.fullmatch(value["model_revision"]):
        raise ExecutionRefusal("model_revision must be an immutable 40-character commit")
    if isinstance(value["site_layer"], bool) or not isinstance(value["site_layer"], int) \
            or not 0 <= value["site_layer"] < 32:
        raise ExecutionRefusal("site_layer must be an integer in [0, 32)")
    quant = _validate_quant(value["quant"])
    timing = _validate_timing(value["timing"])

    if not isinstance(value["output_directory"], str):
        raise ExecutionRefusal("output_directory must be a string path")
    output = Path(value["output_directory"])
    if not output.is_absolute() or str(output.resolve()) != value["output_directory"]:
        raise ExecutionRefusal("output_directory must be an absolute canonical path")
    output = output.resolve()
    if _inside(output, stage.root):
        raise ExecutionRefusal("durable output_directory must be outside the staged repository")
    if output.exists():
        raise ExecutionRefusal("output_directory already exists; overwrite is forbidden")
    if not output.parent.is_dir():
        raise ExecutionRefusal("output_directory parent must already exist")
    if not os.access(output.parent, os.W_OK | os.X_OK):
        raise ExecutionRefusal("output_directory parent is not writable/searchable")

    config = {
        "device": value["device"],
        "model_revision": value["model_revision"],
        "site_layer": value["site_layer"],
        "quant": quant,
        "timing": timing,
        "workload": {"input_shape": [1, D_FF], "input_source": "synthetic-fixed-seed", "seed": 0,
                     "activation_dtype": "float16"},
    }
    return Authorization(value["clearance_id"], value["owner"], _bytes_sha256(clearance_bytes),
                         _canonical_sha256(config), stage, value["device"], output, value["model_revision"],
                         value["site_layer"], quant, timing, value["clock_policy"])


def prepare_invocation(clearance_file: str | os.PathLike[str], *, root: str | os.PathLike[str]) -> Authorization:
    stage = verify_complete_stage(root)
    if _run_git(stage.root, "merge-base", "--is-ancestor", INTAKE_COMMIT, stage.head,
                check=False).returncode != 0:
        raise ExecutionRefusal(f"executed commit must descend from accepted intake {INTAKE_COMMIT}")
    return load_authorization(clearance_file, stage)


def _errors(actual: Iterable[Iterable[float]], expected: Iterable[Iterable[float]]) -> tuple[float, float]:
    squared_error = 0.0
    squared_reference = 0.0
    maximum = 0.0
    for actual_row, expected_row in zip(actual, expected):
        for lhs, rhs in zip(actual_row, expected_row):
            difference = float(lhs) - float(rhs)
            squared_error += difference * difference
            squared_reference += float(rhs) * float(rhs)
            maximum = max(maximum, abs(difference))
    return math.sqrt(squared_error / max(squared_reference, 1e-300)), maximum


class CudaRuntime:
    """Lazy CUDA boundary, constructed only after every static gate passes."""

    def __init__(self, device: str):
        import torch
        import triton

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable after authorization")
        self.torch = torch
        self.triton = triton
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)

    def hardware_identity(self, clock_policy: str) -> dict[str, str]:
        import ctypes

        torch = self.torch
        properties = torch.cuda.get_device_properties(self.device)
        driver_library = ctypes.CDLL("libcuda.so.1")
        driver_version = ctypes.c_int()
        if driver_library.cuDriverGetVersion(ctypes.byref(driver_version)) != 0:
            raise RuntimeError("CUDA driver version query failed")
        if torch.version.cuda is None or not getattr(self.triton, "__version__", None):
            raise RuntimeError("hardware/software identity probe was incomplete")
        return {
            "gpu": f"{properties.name} [{self.device}]",
            "compute_capability": f"{properties.major}.{properties.minor}",
            "driver": f"cuda-driver-api:{driver_version.value}",
            "cuda": str(torch.version.cuda),
            "torch": str(torch.__version__),
            "triton": str(self.triton.__version__),
            "clock_policy": f"owner-recorded:{clock_policy}",
        }

    def fixed_input(self):
        generator = self.torch.Generator(device=self.device)
        generator.manual_seed(0)
        return self.torch.randn((1, D_FF), generator=generator, dtype=self.torch.float16,
                                device=self.device).contiguous()

    def workspace(self, tensor):
        return HFullWorkspace(tensor)

    def quantizer(self, tensor):
        return W4A4ActivationQuantizer(tensor)

    def correctness_metrics(self, tensor, workspace, quantizer) -> dict[str, dict[str, float | bool]]:
        source_rows = tensor.detach().to(dtype=self.torch.float32, device="cpu").tolist()
        results = {}
        for transform_id in ("I", "Hfull"):
            transformed = apply_transform(tensor, transform_id, workspace=workspace)
            observed = transformed.detach().to(dtype=self.torch.float32, device="cpu").tolist()
            expected = forward_rows(source_rows, transform_id)
            transform_relative, transform_max = _errors(observed, expected)
            quantized = quantizer(transformed)
            quantized_rows = quantized.detach().to(dtype=self.torch.float32, device="cpu").tolist()
            expected_quantized, _ = quantize_rows_reference(observed)
            quantizer_relative, quantizer_max = _errors(quantized_rows, expected_quantized)
            difference_squared = sum((lhs - rhs) ** 2 for row, qrow in zip(observed, quantized_rows)
                                     for lhs, rhs in zip(row, qrow))
            reference_squared = sum(value * value for row in observed for value in row)
            epsilon = 1e-12
            results[transform_id] = {
                "passed": (transform_relative <= 5e-3 and transform_max <= 5e-2
                           and quantizer_relative <= 5e-3 and quantizer_max <= 5e-2),
                "transform_relative_error": transform_relative,
                "transform_max_abs_error": transform_max,
                "quantizer_relative_error": quantizer_relative,
                "quantizer_max_abs_error": quantizer_max,
                "local_nmse": difference_squared / max(reference_squared, epsilon),
                "nmse_epsilon": epsilon,
                "activation_absmax": max(abs(value) for row in observed for value in row),
                "activation_rms": math.sqrt(reference_squared / sum(len(row) for row in observed)),
            }
        if not all(result["passed"] for result in results.values()):
            raise RuntimeError(f"GPU transform/quantizer correctness boundary failed: {results}")
        return results


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    location = quantile * (len(sorted_values) - 1)
    lower = int(location)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = location - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def collect_profiles(tensor, workspace, quantizer, timing: dict,
                     *, profile_fn: Callable = profile_transform) -> dict[tuple[str, str], ProfileSummary]:
    summaries = {}
    for timing_identity in ("transform-only", "transform+quantize"):
        for transform_id in ("I", "Hfull"):
            samples = []
            launches = None
            copies = None
            for _ in range(timing["outer_trials"]):
                summary = profile_fn(tensor, transform_id, timing_identity, quantize=quantizer,
                                     workspace=workspace, warmup_ms=timing["warmup_ms"],
                                     repetition_ms=timing["repetition_ms"])
                if summary.timing_identity != timing_identity:
                    raise RuntimeError("profiler returned a mismatched timing identity")
                if launches is not None \
                        and (summary.transform_launches, summary.transform_copies) != (launches, copies):
                    raise RuntimeError("profiler launch/copy observations changed across outer trials")
                launches, copies = summary.transform_launches, summary.transform_copies
                samples.extend(float(sample) for sample in summary.samples_us)
            expected_empty = transform_id == "I" and timing_identity == "transform-only"
            if bool(samples) is expected_empty:
                raise RuntimeError("profiler returned a partial or unexpected empty sample set")
            if any(not math.isfinite(sample) or sample < 0 for sample in samples):
                raise RuntimeError("profiler returned non-finite or negative timing samples")
            ordered = sorted(samples)
            quantiles = (0.0, 0.0, 0.0) if expected_empty else (
                _percentile(ordered, 0.1), _percentile(ordered, 0.5), _percentile(ordered, 0.9))
            summaries[(timing_identity, transform_id)] = ProfileSummary(
                timing_identity, *quantiles, tuple(samples), int(launches), int(copies))
    return summaries


def assemble_records(authorization: Authorization, hardware: dict, oracle: dict,
                     runtime_metrics: dict, profiles: dict) -> tuple[list[dict], list[dict]]:
    record_path = str(authorization.output_directory / RECORD_NAME)
    records = build_plan(head=authorization.stage.head, dirty=False, result_jsonl=record_path)
    raw_records = []
    oracle_passed = all(result["passed"] for result in oracle["results"].values())
    for record in records:
        transform_id = record["transform"]["id"]
        timing_identity = record["timing"]["identity"]
        summary = profiles[(timing_identity, transform_id)]
        oracle_metrics = oracle["results"][transform_id]
        measured = runtime_metrics[transform_id]
        record["run_id"] = f"{RUN_ID}-{authorization.config_sha256[:12]}"
        record["model"]["revision"] = authorization.model_revision
        record["quant"] = copy.deepcopy(authorization.quant)
        record["site"]["layer"] = authorization.site_layer
        record["hardware"] = copy.deepcopy(hardware)
        record["timing"].update(authorization.timing)
        record["metrics"].update({
            "correctness_passed": bool(oracle_passed and measured["passed"]),
            "inverse_rel_error": oracle_metrics["inverse_relative_error"],
            "inverse_max_abs_error": oracle_metrics["inverse_max_abs_error"],
            "local_equivalence_rel_error": max(oracle_metrics["local_equivalence_relative_error"],
                                               measured["transform_relative_error"]),
            "local_equivalence_max_abs_error": max(oracle_metrics["local_equivalence_max_abs_error"],
                                                    measured["transform_max_abs_error"]),
            "local_nmse": measured["local_nmse"],
            "nmse_epsilon": measured["nmse_epsilon"],
            "activation_absmax": measured["activation_absmax"],
            "activation_rms": measured["activation_rms"],
        })
        metric_prefix = "rotation_us" if timing_identity == "transform-only" else "rotation_quantize_us"
        record["metrics"].update({
            f"{metric_prefix}_p10": summary.p10_us,
            f"{metric_prefix}_median": summary.median_us,
            f"{metric_prefix}_p90": summary.p90_us,
        })
        quantizer_launches = quantizer_copies = 0
        if timing_identity == "transform+quantize":
            quantizer_launches = W4A4ActivationQuantizer.launches_per_call
            quantizer_copies = W4A4ActivationQuantizer.copies_per_call
        record["execution"].update({
            "status": "measurement",
            "scheduler_clearance": True,
            "scientific_evidence": False,
            "synthetic_input": True,
            "transform_launches": summary.transform_launches,
            "transform_copies": summary.transform_copies,
            "total_launches": summary.transform_launches + quantizer_launches,
        })
        if quantizer_copies:
            raise RuntimeError("quantizer unexpectedly declared hidden copies")
        record["notes"] = (
            "SYNTHETIC FIXED-SEED NON-MODEL/NON-PPL MICROPROFILE; not model quality, perplexity, or "
            f"end-to-end evidence; clearance_id={authorization.clearance_id}; "
            f"owner={authorization.owner}; driver_commit={authorization.stage.head}; "
            f"execution_config_sha256={authorization.config_sha256}; fusion=none; "
            f"a4_reference_rel_error={measured['quantizer_relative_error']}; "
            f"a4_reference_max_abs_error={measured['quantizer_max_abs_error']}; "
            "transform+quantize is sequential transform-then-quantize."
        )
        raw_records.append({
            "schema_version": RAW_SCHEMA_VERSION,
            "run_id": record["run_id"],
            "experiment_commit": authorization.stage.head,
            "transform": transform_id,
            "timing_identity": timing_identity,
            "unit": "us",
            "samples": list(summary.samples_us),
            "synthetic_non_model": True,
        })
    validate_records(records)
    validate_raw_records(raw_records)
    return records, raw_records


def validate_raw_records(records: object) -> list[dict]:
    if not isinstance(records, list) or len(records) != 4:
        raise ExecutionRefusal("raw timing artifact must contain exactly four rows")
    seen = set()
    for record in records:
        expected = {"schema_version", "run_id", "experiment_commit", "transform", "timing_identity",
                    "unit", "samples", "synthetic_non_model"}
        if not isinstance(record, dict):
            raise ExecutionRefusal("raw timing row must be an object")
        _exact_keys(record, expected, "raw timing row")
        if record["schema_version"] != RAW_SCHEMA_VERSION or record["unit"] != "us" \
                or record["synthetic_non_model"] is not True:
            raise ExecutionRefusal("raw timing identity is invalid")
        if record["transform"] not in ("I", "Hfull") \
                or record["timing_identity"] not in ("transform-only", "transform+quantize"):
            raise ExecutionRefusal("raw timing row is outside the accepted matrix")
        if not _HEX40.fullmatch(record["experiment_commit"]):
            raise ExecutionRefusal("raw timing commit is invalid")
        if not isinstance(record["run_id"], str) or not record["run_id"]:
            raise ExecutionRefusal("raw timing run_id must be a non-empty string")
        if not isinstance(record["samples"], list) or any(
                isinstance(sample, bool) or not isinstance(sample, (int, float))
                or not math.isfinite(sample) or sample < 0 for sample in record["samples"]):
            raise ExecutionRefusal("raw timing samples must be finite nonnegative numbers")
        key = (record["transform"], record["timing_identity"])
        if key in seen:
            raise ExecutionRefusal("raw timing artifact contains a duplicate logical row")
        seen.add(key)
        empty_expected = key == ("I", "transform-only")
        if bool(record["samples"]) is empty_expected:
            raise ExecutionRefusal("raw timing artifact contains a partial sample row")
    return records


def _jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(row, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
                   for row in rows)


def _write_file(path: Path, value: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_output(authorization: Authorization, records: list[dict], raw_records: list[dict]) -> None:
    record_text = dumps_jsonl(records)
    if loads_jsonl(record_text) != records:
        raise RuntimeError("validated record JSONL did not round trip")
    validate_raw_records(raw_records)
    raw_text = _jsonl(raw_records)
    output = authorization.output_directory
    try:
        output.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise ExecutionRefusal("output_directory appeared before commit; overwrite is forbidden") from exc
    try:
        output_manifest = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "source_commit": authorization.stage.head,
            "driver_commit": authorization.stage.head,
            "transform_commit": authorization.stage.head,
            "stage_metadata_sha256": authorization.stage.metadata_sha256,
            "stage_manifest_sha256": authorization.stage.manifest_sha256,
            "clearance_id": authorization.clearance_id,
            "clearance_sha256": authorization.clearance_sha256,
            "execution_config_sha256": authorization.config_sha256,
            "record_jsonl": RECORD_NAME,
            "record_jsonl_sha256": _bytes_sha256(record_text.encode("utf-8")),
            "raw_samples_jsonl": RAW_NAME,
            "raw_samples_jsonl_sha256": _bytes_sha256(raw_text.encode("utf-8")),
            "synthetic_non_model_non_ppl": True,
            "scientific_evidence": False,
        }
        _write_file(output / RAW_NAME, raw_text.encode("utf-8"))
        _write_file(output / OUTPUT_MANIFEST_NAME,
                    (json.dumps(output_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
        _write_file(output / RECORD_NAME, record_text.encode("utf-8"))
        directory_descriptor = os.open(output, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        for name in (RAW_NAME, OUTPUT_MANIFEST_NAME, RECORD_NAME):
            (output / name).unlink(missing_ok=True)
        for child in output.iterdir():
            if child.name.startswith(".") and child.name.endswith(".tmp"):
                child.unlink(missing_ok=True)
        output.rmdir()
        raise


def execute(authorization: Authorization, *, runtime_factory: Callable = CudaRuntime,
            profile_fn: Callable = profile_transform, oracle_fn: Callable = run_oracle) -> None:
    if verify_complete_stage(authorization.stage.root) != authorization.stage:
        raise ExecutionRefusal("stage identity changed after scheduler clearance validation")
    if authorization.output_directory.exists():
        raise ExecutionRefusal("output_directory appeared before execution; overwrite is forbidden")
    oracle = oracle_fn(seed=0, token_rows=1, weight_rows=2)
    if not all(result["passed"] for result in oracle["results"].values()):
        raise RuntimeError("existing correctness boundary did not pass")
    if verify_complete_stage(authorization.stage.root) != authorization.stage:
        raise ExecutionRefusal("stage identity changed during the pre-GPU correctness boundary")
    runtime = runtime_factory(authorization.device)
    hardware = runtime.hardware_identity(authorization.clock_policy)
    tensor = runtime.fixed_input()
    workspace = runtime.workspace(tensor)
    quantizer = runtime.quantizer(tensor)
    runtime_metrics = runtime.correctness_metrics(tensor, workspace, quantizer)
    profiles = collect_profiles(tensor, workspace, quantizer, authorization.timing, profile_fn=profile_fn)
    records, raw_records = assemble_records(authorization, hardware, oracle, runtime_metrics, profiles)
    write_output(authorization, records, raw_records)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler-clearance-file", required=True,
                        help="absolute owner-only phase-a-scheduler-clearance-v1 JSON file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        authorization = prepare_invocation(args.scheduler_clearance_file, root=Path.cwd())
        execute(authorization)
    except (ExecutionRefusal, OSError, RuntimeError, ValueError, AssertionError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(authorization.output_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
