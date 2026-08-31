"""Submission-free Phase A matrix preflight.

This module never imports Torch, Triton, CUDA bindings, or scheduler tooling.
Its execution switch is intentionally fail-closed because this task has no
scheduler clearance.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import PurePosixPath

from .reference import D_FF, D_MODEL, transform_spec
from .schema import BASE_COMMIT, ORACLE_COMMIT, SCHEMA_VERSION, dumps_jsonl, validate_records

SOURCE_BRANCHES = (
    "fm/structured-hadamard-phase-a",
    "fm/structured-hadamard-phase-a-exec-r1",
    "fm/structured-hadamard-phase-a-kernel-profile-r1",
)
ACTIVE_ROOT = "/data/scratch-fast/kwen1"
DEFAULT_RESULT_JSONL = f"{ACTIVE_ROOT}/structured-hadamard/phase-a/results/phase-a.jsonl"
RUN_ID = "phase-a-llama2-7b-decode-seed0"


def _git(*args: str) -> str:
    completed = subprocess.run(("git", *args), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return completed.stdout.strip()


def discover_code_identity() -> dict:
    branch = _git("branch", "--show-current")
    if branch not in SOURCE_BRANCHES:
        raise RuntimeError(f"preflight requires one of {SOURCE_BRANCHES!r}, got {branch!r}")
    head = _git("rev-parse", "HEAD")
    ancestor = subprocess.run(("git", "merge-base", "--is-ancestor", BASE_COMMIT, head), check=False)
    if ancestor.returncode != 0:
        raise RuntimeError(f"pinned base {BASE_COMMIT} is not an ancestor of {head}")
    return {"head": head, "dirty": bool(_git("status", "--porcelain"))}


def _empty_metrics() -> dict:
    return {
        "correctness_passed": None,
        "inverse_rel_error": None,
        "inverse_max_abs_error": None,
        "local_equivalence_rel_error": None,
        "local_equivalence_max_abs_error": None,
        "local_nmse": None,
        "nmse_epsilon": None,
        "activation_absmax": None,
        "activation_rms": None,
        "ppl_wikitext2": None,
        "rotation_us_p10": None,
        "rotation_us_median": None,
        "rotation_us_p90": None,
        "rotation_quantize_us_p10": None,
        "rotation_quantize_us_median": None,
        "rotation_quantize_us_p90": None,
        "affected_layer_us_median": None,
        "end_to_end_ms_per_token": None,
        "tokens_per_s": None,
    }


def _raw_samples_path(record_jsonl: str) -> str:
    path = PurePosixPath(record_jsonl)
    return str(path.with_name(path.name.removesuffix(".jsonl") + ".raw-samples.jsonl"))


def build_plan(*, head: str, dirty: bool, result_jsonl: str = DEFAULT_RESULT_JSONL) -> list[dict]:
    if not PurePosixPath(result_jsonl).is_absolute() or not result_jsonl.endswith(".jsonl"):
        raise ValueError("result_jsonl must be an absolute .jsonl path")
    template = {
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "code": {
            "experiment_commit": head,
            "base_commit": BASE_COMMIT,
            "transform_impl_commit": head,
            "origin": "https://github.com/kw7243/triton.git",
            "intended_upstream": "https://github.com/triton-lang/triton.git",
            "license": "MIT",
            "oracle_repo": "https://github.com/kw7243/QuaRot.git",
            "oracle_upstream": "https://github.com/spcl/QuaRot.git",
            "oracle_commit": ORACLE_COMMIT,
            "oracle_license": "Apache-2.0",
            "dirty": dirty,
        },
        "model": {
            "id": "meta-llama/Llama-2-7b-hf",
            "revision": "UNRESOLVED-NO-MODEL-DOWNLOAD",
            "d_model": D_MODEL,
            "d_ff": D_FF,
            "n_layers": 32,
        },
        "quant": {
            "w_bits": 4,
            "a_bits": 4,
            "w_group_size": "UNRESOLVED-PHASE-A-OWNER",
            "a_group_size": "per-row",
            "w_symmetric": True,
            "a_symmetric": True,
            "scale_granularity": "dynamic-per-row-A4;W4-UNRESOLVED",
            "clip": "UNRESOLVED-PHASE-A-OWNER",
            "calibration_dataset": "UNRESOLVED-NO-MODEL-DOWNLOAD",
            "calibration_seed": 0,
            "calibration_rows": 8192,
        },
        "site": {"layer": 0, "kind": "mlp.down_proj.input"},
        "transform": {},
        "workload": {
            "mode": "decode",
            "input_shape": [1, D_FF],
            "weight_shape": [D_MODEL, D_FF],
            "batch": 1,
            "prompt_tokens": 2048,
            "output_tokens": 128,
            "activation_dtype": "float16",
            "contiguous": True,
            "input_source": "synthetic-fixed-seed",
            "seed": 0,
        },
        "hardware": {key: "UNEXECUTED" for key in
                     ("gpu", "compute_capability", "driver", "cuda", "torch", "triton", "clock_policy")},
        "metrics": _empty_metrics(),
        "timing": {},
        "execution": {
            "status": "unexecuted-plan",
            "scheduler_clearance": False,
            "scientific_evidence": False,
            "synthetic_input": True,
            "transform_launches": None,
            "transform_copies": None,
            "total_launches": None,
        },
        "artifacts": {"record_jsonl": result_jsonl, "raw_samples_jsonl": _raw_samples_path(result_jsonl)},
        "notes": "UNEXECUTED PLAN ONLY; synthetic inputs; no GPU, CUDA, scheduler, benchmark, or scientific evidence.",
    }

    records = []
    for timing_identity in ("transform-only", "transform+quantize"):
        for transform_id in ("I", "Hfull"):
            record = copy.deepcopy(template)
            spec = transform_spec(transform_id)
            record["transform"] = {
                "id": spec.id,
                "axis": spec.axis,
                "d": spec.d,
                "block_size": spec.block_size,
                "K": spec.K,
                "q": spec.q,
                "normalization": spec.normalization,
                "channel_order": spec.channel_order,
                "sign": spec.sign,
                "permutation": spec.permutation,
                "matrix_digest": spec.matrix_digest,
                "fusion": "none",
                "implementation": spec.implementation,
            }
            record["timing"] = {
                "identity": timing_identity,
                "composition": ("transform-only" if timing_identity == "transform-only"
                                else "sequential-transform-then-quantize"),
                "warmup_ms": 25,
                "repetition_ms": 200,
                "outer_trials": 5,
                "timer": "device-events",
                "synchronized": True,
                "statistics": "p10,p50,p90",
            }
            records.append(record)
    return validate_records(records)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print the no-CUDA Phase A I/Hfull matrix")
    parser.add_argument("--result-jsonl", default=DEFAULT_RESULT_JSONL,
                        help="absolute future result path recorded in every planned row")
    parser.add_argument("--scheduler-clearance", choices=("false", ), default="false",
                        help="hard safety gate; this preparation revision accepts only false")
    parser.add_argument("--execute", action="store_true", help="request execution (always refused in this revision)")
    parser.add_argument("--format", choices=("jsonl", "pretty"), default="pretty")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.execute:
        print("REFUSED: scheduler_clearance=false; this preparation CLI cannot execute or submit work.", file=sys.stderr)
        return 2
    identity = discover_code_identity()
    records = build_plan(result_jsonl=args.result_jsonl, head=identity["head"], dirty=identity["dirty"])
    if args.format == "jsonl":
        sys.stdout.write(dumps_jsonl(records))
    else:
        print(json.dumps({
            "kind": "UNEXECUTED-PHASE-A-PLAN",
            "scheduler_clearance": False,
            "scientific_evidence": False,
            "active_root": ACTIVE_ROOT,
            "result_jsonl": args.result_jsonl,
            "matrix": records,
        }, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
