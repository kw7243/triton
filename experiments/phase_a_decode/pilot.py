"""Bounded two-row performance pilot for the Phase A J/F/H decoder.

This module adds no kernels. It reuses the correctness, tuning, timing, and
artifact functions in :mod:`benchmark` while making the pilot's two-row
scientific decision contract explicit and fail closed.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shlex
import shutil
import socket
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PROTOCOL = "phase-a-hurwitz-two-row-pilot-v1"
PILOT_S = (192, )
PILOT_T_KV = (4096, 32768)
SHORT_CONTEXT_REGRESSION = 1.05
PRIMARY_GO_SPEEDUP = 1.25
PRIMARY_OPTIMIZE_SPEEDUP = 1.10
MAX_HOT_STABILITY = 0.05

SCIENTIFIC_ARTIFACTS = (
    "correctness.json",
    "tuning.json",
    "trial_timings.json",
    "timings.csv",
    "README.md",
    "jh_speedup.png",
)
ENVELOPE_ARTIFACTS = (
    "REPRODUCIBILITY_METADATA.json",
    "staged_snapshot.txt",
    "command.txt",
    "environment.lock.txt",
    "system.txt",
    "run_metadata.json",
)


@dataclass(frozen=True)
class Runtime:
    benchmark: Any
    torch: Any
    triton: Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--s", type=int, nargs="+", default=list(PILOT_S))
    parser.add_argument("--t-kv", type=int, nargs="+", default=list(PILOT_T_KV))
    parser.add_argument("--roles", type=int, default=2)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=200)
    parser.add_argument("--outer-trials", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    expected = {
        "seed": 0,
        "s": list(PILOT_S),
        "t_kv": list(PILOT_T_KV),
        "roles": 2,
        "kv_heads": 8,
        "head_dim": 128,
        "warmup": 25,
        "rep": 200,
        "outer_trials": 5,
    }
    actual = {name: getattr(args, name) for name in expected}
    if actual != expected:
        raise ValueError(f"arguments do not match the declared two-row pilot protocol: {actual!r}")
    if not args.output.is_absolute():
        raise ValueError("--output must be an absolute path outside the staged repository")


def _rows_by_tkv(rows: Sequence[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    pairs = [(row.get("S"), row.get("Tkv")) for row in rows]
    if pairs != [(192, 4096), (192, 32768)]:
        raise ValueError(f"pilot rows must be exactly [(192, 4096), (192, 32768)], got {pairs!r}")
    return {row["Tkv"]: row for row in rows}


def decide_pilot(rows: Sequence[dict[str, Any]]) -> str:
    """Apply the retained thresholds after requiring both pilot rows stable."""

    by_tkv = _rows_by_tkv(rows)
    for row in rows:
        if not row.get("stable", False):
            row["decision"] = "UNSTABLE — DO NOT INTERPRET"

    if not all(row.get("stable", False) for row in rows):
        return "UNSTABLE — DO NOT INTERPRET"

    short = by_tkv[4096]
    primary = by_tkv[32768]
    if short["h_hot_p50_ms"] > SHORT_CONTEXT_REGRESSION * short["j_hot_p50_ms"]:
        short["decision"] = "KILL/RETARGET — >5% SHORT REGRESSION"
        primary["decision"] = "NOT INTERPRETED — SHORT REGRESSION"
        return "KILL/RETARGET"

    short["decision"] = "SHORT-CONTEXT PASS"
    speedup = primary["jh_hot_speedup"]
    if speedup >= PRIMARY_GO_SPEEDUP:
        primary["decision"] = "GO"
        return "GO"
    if speedup >= PRIMARY_OPTIMIZE_SPEEDUP:
        primary["decision"] = "OPTIMIZE ONCE"
        return "OPTIMIZE ONCE"
    primary["decision"] = "KILL"
    return "KILL"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_runtime() -> Runtime:
    import torch
    import triton

    from experiments.phase_a_decode import benchmark

    return Runtime(benchmark=benchmark, torch=torch, triton=triton)


def discover_stage(environ: Mapping[str, str] | None = None, cwd: Path | None = None) -> dict[str, Any]:
    environ = os.environ if environ is None else environ
    stage_value = environ.get("RESEARCH_REPRO_STAGED_DIR")
    source_value = environ.get("RESEARCH_REPRO_SOURCE_REPO") or environ.get("SOURCE_REPO")
    if not stage_value or not source_value:
        raise RuntimeError("RESEARCH_REPRO_STAGED_DIR and RESEARCH_REPRO_SOURCE_REPO are required")

    stage = Path(stage_value).resolve(strict=True)
    source = Path(source_value).resolve(strict=True)
    current = Path.cwd().resolve() if cwd is None else cwd.resolve()
    if current != stage:
        raise RuntimeError(f"pilot must run from the staged repository: cwd={current}, stage={stage}")
    if source == stage:
        raise RuntimeError("refusing to run the pilot from the mutable source repository")
    metadata = stage / "REPRODUCIBILITY_METADATA.json"
    if not metadata.is_file():
        raise RuntimeError(f"staged repository metadata is missing: {metadata}")
    return {"stage": stage, "source": source, "metadata": metadata}


def _environment_lock() -> str:
    packages = sorted((distribution.metadata.get("Name", "unknown"), distribution.version)
                      for distribution in importlib.metadata.distributions())
    return "".join(f"{name}=={version}\n" for name, version in packages)


def _system_record(runtime: Runtime, provenance: Mapping[str, Any]) -> str:
    torch = runtime.torch
    properties = torch.cuda.get_device_properties(0)
    free_bytes, visible_bytes = torch.cuda.mem_get_info(0)
    fields = {
        "hostname": socket.getfqdn(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "unset"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "source_repo": str(provenance["source"]),
        "staged_repo": str(provenance["stage"]),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "triton": runtime.triton.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": torch.cuda.get_device_capability(0),
        "vram_total_bytes": properties.total_memory,
        "vram_free_bytes": free_bytes,
        "vram_visible_total_bytes": visible_bytes,
        "bf16_supported": torch.cuda.is_bf16_supported(),
    }
    return "".join(f"{key}={value}\n" for key, value in fields.items())


def _protocol_metadata() -> dict[str, Any]:
    return {
        "id": PROTOCOL,
        "rows": [{"S": 192, "Tkv": 4096}, {"S": 192, "Tkv": 32768}],
        "timing": {"warmup_ms": 25, "initial_rep_ms": 200, "max_rep_ms": 1600,
                   "outer_trials": 5, "quantiles": [0.2, 0.5, 0.8]},
        "thresholds": {"both_rows_max_hot_stability": MAX_HOT_STABILITY,
                       "short_context_h_over_j_max": SHORT_CONTEXT_REGRESSION,
                       "primary_go_j_over_h_min": PRIMARY_GO_SPEEDUP,
                       "primary_optimize_j_over_h_min": PRIMARY_OPTIMIZE_SPEEDUP},
    }


def _validate_row_schema(rows: Sequence[dict[str, Any]]) -> None:
    _rows_by_tkv(rows)
    required = {"S", "Tkv", "roles", "kv_heads", "head_dim", "chunks", "output_bytes",
                "warmup_ms", "rep_ms", "outer_trials", "j_config", "f_config", "h_config",
                "jh_hot_speedup", "jh_cold_speedup", "stable", "h_max_abs", "h_relative_fro"}
    for variant in ("j", "f", "h"):
        for mode in ("cold", "hot"):
            required.update({f"{variant}_{mode}_p20_ms", f"{variant}_{mode}_p50_ms",
                             f"{variant}_{mode}_p80_ms", f"{variant}_{mode}_gchunks_s",
                             f"{variant}_{mode}_output_gib_s", f"{variant}_{mode}_stability"})
    for index, row in enumerate(rows):
        missing = sorted(required.difference(row))
        if missing:
            raise RuntimeError(f"pilot row {index} is missing required artifact fields: {missing}")


def _verify_artifacts(output: Path) -> None:
    missing = [name for name in SCIENTIFIC_ARTIFACTS if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"scientific artifact write was incomplete: {missing}")


def execute_pilot(args: argparse.Namespace, runtime: Runtime, provenance: Mapping[str, Any],
                  command: str) -> str:
    validate_args(args)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(provenance["metadata"], args.output / "REPRODUCIBILITY_METADATA.json")
    _atomic_write_text(args.output / "staged_snapshot.txt", f"{provenance['stage']}\n")
    _atomic_write_text(args.output / "command.txt", command + "\n")
    _atomic_write_text(args.output / "environment.lock.txt", _environment_lock())

    metadata: dict[str, Any] = {
        "status": "running",
        "phase": "runtime-preflight",
        "protocol": _protocol_metadata(),
        "source_repo": str(provenance["source"]),
        "staged_repo": str(provenance["stage"]),
    }
    metadata_path = args.output / "run_metadata.json"
    _atomic_write_json(metadata_path, metadata)

    try:
        torch = runtime.torch
        benchmark = runtime.benchmark
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; run only through a staged Slurm GPU allocation")

        metadata.update({
            "git_commit": benchmark.git("rev-parse", "HEAD"),
            "git_branch": benchmark.git("branch", "--show-current"),
            "cwd": os.getcwd(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "triton": runtime.triton.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "compute_capability": torch.cuda.get_device_capability(0),
            "assumptions": {"quaternion": "scalar-first (w,x,y,z)", "id": "p*S+s"},
        })
        _atomic_write_text(args.output / "system.txt", _system_record(runtime, provenance))

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        metadata.update(phase="correctness")
        _atomic_write_json(metadata_path, metadata)
        print("correctness: starting S=192", flush=True)
        tables, correctness, bf16 = benchmark.run_correctness(args)
        if 192 not in tables:
            raise RuntimeError("correctness did not return the required fp16 S=192 tables")
        print(f"correctness: passed; bf16: {bf16}", flush=True)

        metadata.update(phase="tuning")
        _atomic_write_json(metadata_path, metadata)
        selected, scores = benchmark.tune(args, 192, tables[192])
        tuning = {"configs": [config[2] for config in benchmark.CONFIGS],
                  "selection_shape_tkv": 4096,
                  "by_s": {"192": {"scores_ms": scores,
                                     "selected": {variant: selected[variant][2]
                                                  for variant in benchmark.VARIANTS}}}}

        metadata.update(phase="timing")
        _atomic_write_json(metadata_path, metadata)
        summary = benchmark.correctness_summary(correctness, 192)
        rows, raw = [], {}
        for t_kv in PILOT_T_KV:
            print(f"timing: S=192, Tkv={t_kv}", flush=True)
            row, attempts = benchmark.measure(args, 192, t_kv, tables[192], selected)
            row.update(summary)
            row["h_max_abs"] = max(row["h_max_abs_vs_j"], row["h_max_abs_vs_oracle"])
            row["h_relative_fro"] = max(row["h_relative_fro_vs_j"],
                                         row["h_relative_fro_vs_oracle"])
            rows.append(row)
            raw[f"S192-T{t_kv}"] = attempts

        _validate_row_schema(rows)
        conclusion = decide_pilot(rows)
        benchmark.write_artifacts(args.output, rows, raw, correctness, bf16, tuning, conclusion)
        _verify_artifacts(args.output)

        metadata.update(status="passed", phase="terminal", conclusion=conclusion,
                        artifact_schema={"envelope": list(ENVELOPE_ARTIFACTS),
                                         "scientific": list(SCIENTIFIC_ARTIFACTS)})
        _atomic_write_json(metadata_path, metadata)
        print(f"conclusion: {conclusion}", flush=True)
        return conclusion
    except BaseException as exc:
        metadata.update(status="failed", phase="terminal", error_type=type(exc).__name__,
                        error=str(exc))
        _atomic_write_json(metadata_path, metadata)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    provenance = discover_stage()
    runtime = _load_runtime()
    command = shlex.join([sys.executable, *sys.argv])
    execute_pilot(args, runtime, provenance, command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
