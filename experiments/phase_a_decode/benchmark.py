"""Independent paper-spec Phase A J/F/H decode benchmark.

Quaternion order is scalar-first ``(w,x,y,z)`` and ``id=p*S+s``. No HQMQ
implementation source was used; the surrounding Triton tree retains its MIT license.
"""

import argparse
import csv
import json
import math
import os
import random
import re
import statistics
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import triton
import triton.language as tl

from gpu_correctness.protocol import (
    ABS_TOL,
    CONFIGS,
    INPUT_KINDS,
    PRIMARY_UNITS,
    RANDOM_IDS_PER_ROLE_HEAD,
    REL_TOL,
    SECONDARY_SIZES,
    CorrectnessArguments,
    ScientificComparisonFailure,
    contract_snapshot,
    validate_correctness_arguments,
)
from gpu_correctness.result_protocol import (
    STATE_FILE,
    atomic_write_json,
    atomic_write_text,
    read_json,
    record_boundary,
)

Q = (0.2, 0.5, 0.8)
VARIANTS = ("J", "F", "H")
MODE = {"J": 0, "F": 1, "H": 2}


@triton.jit
def decode_kernel(ids, joint, secondary, primary, out, n, chunks_per_head, S: tl.constexpr,
                  MODE: tl.constexpr, BLOCK: tl.constexpr):
    c = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = c < n
    code = tl.load(ids + c, mask=mask, other=0)
    rh = c // chunks_per_head
    out_off = c * 4
    if MODE == 0:  # J: one irregular joint-table gather.
        off = (rh * (24 * S) + code) * 4
        for k in range(4):
            tl.store(out + out_off + k, tl.load(joint + off + k, mask=mask), mask=mask)
    else:
        p, s = code // S, code % S
        off = (rh * S + s) * 4
        w = tl.load(secondary + off + 0, mask=mask).to(tl.float32)
        x = tl.load(secondary + off + 1, mask=mask).to(tl.float32)
        y = tl.load(secondary + off + 2, mask=mask).to(tl.float32)
        z = tl.load(secondary + off + 3, mask=mask).to(tl.float32)
        if MODE == 1:  # F: primary gather plus generic Hamilton product.
            poff = p * 4
            pw = tl.load(primary + poff + 0, mask=mask).to(tl.float32)
            px = tl.load(primary + poff + 1, mask=mask).to(tl.float32)
            py = tl.load(primary + poff + 2, mask=mask).to(tl.float32)
            pz = tl.load(primary + poff + 3, mask=mask).to(tl.float32)
            ow = pw * w - px * x - py * y - pz * z
            ox = pw * x + px * w + py * z - pz * y
            oy = pw * y - px * z + py * w + pz * x
            oz = pw * z + px * y - py * x + pz * w
        else:  # H: axis signed permutations or half-unit signed sums.
            aw = tl.where(p == 0, w, tl.where(p == 1, -w, tl.where(p == 2, -x, tl.where(
                p == 3, x, tl.where(p == 4, -y, tl.where(p == 5, y, tl.where(p == 6, -z, z)))))))
            ax = tl.where(p == 0, x, tl.where(p == 1, -x, tl.where(p == 2, w, tl.where(
                p == 3, -w, tl.where(p == 4, z, tl.where(p == 5, -z, tl.where(p == 6, -y, y)))))))
            ay = tl.where(p == 0, y, tl.where(p == 1, -y, tl.where(p == 2, -z, tl.where(
                p == 3, z, tl.where(p == 4, w, tl.where(p == 5, -w, tl.where(p == 6, x, -x)))))))
            az = tl.where(p == 0, z, tl.where(p == 1, -z, tl.where(p == 2, y, tl.where(
                p == 3, -y, tl.where(p == 4, -x, tl.where(p == 5, x, tl.where(p == 6, w, -w)))))))
            h = p - 8
            sw = tl.where((h & 1) == 0, 1.0, -1.0)
            sx = tl.where((h & 2) == 0, 1.0, -1.0)
            sy = tl.where((h & 4) == 0, 1.0, -1.0)
            sz = tl.where((h & 8) == 0, 1.0, -1.0)
            hw = 0.5 * (sw * w - sx * x - sy * y - sz * z)
            hx = 0.5 * (sw * x + sx * w + sy * z - sz * y)
            hy = 0.5 * (sw * y - sx * z + sy * w + sz * x)
            hz = 0.5 * (sw * z + sx * y - sy * x + sz * w)
            axis = p < 8
            ow, ox = tl.where(axis, aw, hw), tl.where(axis, ax, hx)
            oy, oz = tl.where(axis, ay, hy), tl.where(axis, az, hz)
        tl.store(out + out_off + 0, ow, mask=mask)
        tl.store(out + out_off + 1, ox, mask=mask)
        tl.store(out + out_off + 2, oy, mask=mask)
        tl.store(out + out_off + 3, oz, mask=mask)


def primary_units():
    units = [(1, 0, 0, 0), (-1, 0, 0, 0), (0, 1, 0, 0), (0, -1, 0, 0),
             (0, 0, 1, 0), (0, 0, -1, 0), (0, 0, 0, 1), (0, 0, 0, -1)]
    for bits in range(16):
        units.append(tuple(0.5 if not bits & (1 << k) else -0.5 for k in range(4)))
    return torch.tensor(units, dtype=torch.float32)


def hamilton(a, b):
    """Independent scalar-first float32 PyTorch oracle."""
    aw, ax, ay, az = a.float().unbind(-1)
    bw, bx, by, bz = b.float().unbind(-1)
    return torch.stack((aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw), -1)


def make_tables(seed, roles, heads, s_size, dtype):
    gen = torch.Generator().manual_seed(seed)
    sec = torch.randn((roles, heads, s_size, 4), generator=gen)
    sec /= torch.linalg.vector_norm(sec, dim=-1, keepdim=True).clamp_min(1e-12)
    sec = sec.to("cuda", dtype=dtype)
    p32 = primary_units().cuda()
    joint = hamilton(p32.view(1, 1, 24, 1, 4),
                      sec.float().view(roles, heads, 1, s_size, 4))
    return {"joint": joint.reshape(roles, heads, 24 * s_size, 4).to(dtype),
            "secondary": sec, "primary": p32.to(dtype), "primary32": p32}


def launch(variant, ids, tables, out, chunks_per_head, s_size, config):
    block, warps, _ = config
    decode_kernel[(triton.cdiv(ids.numel(), block),)](
        ids, tables["joint"], tables["secondary"], tables["primary"], out,
        ids.numel(), chunks_per_head, S=s_size, MODE=MODE[variant], BLOCK=block,
        num_warps=warps)


def error(actual, reference):
    delta = actual.float() - reference.float()
    return {"max_abs": delta.abs().max().item(),
            "relative_fro": (torch.linalg.vector_norm(delta) /
                             torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)).item()}


def require_close(check, label, metric):
    if metric["max_abs"] > ABS_TOL or metric["relative_fro"] > REL_TOL:
        raise ScientificComparisonFailure(check, f"{label}: {metric}")


def correctness_case(seed, roles, heads, s_size, dtype, tables, kind, config, progress):
    rh_count = roles * heads
    if kind == "exhaustive":
        ids2 = torch.arange(24 * s_size, dtype=torch.int32).repeat(rh_count, 1)
    else:
        ids2 = torch.randint(
            24 * s_size,
            (rh_count, RANDOM_IDS_PER_ROLE_HEAD),
            dtype=torch.int32,
            generator=torch.Generator().manual_seed(seed),
        )
    ids2 = ids2.cuda()
    per_head, ids = ids2.shape[1], ids2.flatten()
    rh = torch.arange(rh_count, device="cuda")[:, None].expand_as(ids2).flatten()
    code, p = ids.long(), ids.long() // s_size
    s = code % s_size
    gathered = tables["joint"].view(rh_count, 24 * s_size, 4)[rh, code]
    oracle = hamilton(tables["primary32"][p],
                       tables["secondary"].view(rh_count, s_size, 4)[rh, s])
    outputs = {}
    progress("kernel_compile_or_launch", dtype=str(dtype), S=s_size, input=kind, config=config[2])
    for variant in VARIANTS:
        outputs[variant] = torch.empty((ids.numel(), 4), dtype=dtype, device="cuda")
        launch(variant, ids, tables, outputs[variant], per_head, s_size, config)
    progress("cuda_synchronize", dtype=str(dtype), S=s_size, input=kind, config=config[2])
    torch.cuda.synchronize()
    progress("inside_comparison", dtype=str(dtype), S=s_size, input=kind, config=config[2])
    if not torch.equal(outputs["J"], gathered):
        raise ScientificComparisonFailure(
            "J_bitwise_gather", "J is not bitwise equal to the independent PyTorch gather"
        )
    for variant in VARIANTS:
        if not torch.isfinite(outputs[variant]).all():
            raise ScientificComparisonFailure("finite", f"{variant} contains NaN/Inf")
    for variant in ("F", "H"):
        if not torch.equal(outputs[variant][p < 8], outputs["J"][p < 8]):
            raise ScientificComparisonFailure(
                "F_H_axis_bitwise_J", f"{variant} axis results are not bitwise equal to J"
            )
    metrics = {}
    for variant in ("F", "H"):
        metrics[f"{variant}_vs_J"] = error(outputs[variant], outputs["J"])
        metrics[f"{variant}_vs_oracle"] = error(outputs[variant], oracle)
        metrics[f"{variant}_vs_quantized_oracle"] = error(outputs[variant], oracle.to(dtype))
        require_close("F_H_vs_J_tolerance", f"{variant} vs J", metrics[f"{variant}_vs_J"])
        require_close(
            "F_H_vs_quantized_oracle_tolerance",
            f"{variant} vs quantized oracle",
            metrics[f"{variant}_vs_quantized_oracle"],
        )
        if dtype == torch.float16:
            require_close(
                "F_H_vs_float32_oracle_tolerance",
                f"{variant} vs float32 oracle",
                metrics[f"{variant}_vs_oracle"],
            )
    return {
        "dtype": str(dtype).split(".")[-1],
        "S": s_size,
        "input": kind,
        "config": config[2],
        "chunks": ids.numel(),
        "checks": {
            "finite": True,
            "J_bitwise_gather": True,
            "F_H_axis_bitwise_J": True,
            "F_H_vs_J_tolerance": True,
            "F_H_vs_quantized_oracle_tolerance": True,
            "F_H_vs_float32_oracle_tolerance": dtype == torch.float16,
        },
        "metrics": metrics,
    }


def run_correctness(args, progress):
    tables, records = {}, []
    dtypes = [torch.float16]
    bf16_supported = torch.cuda.is_bf16_supported()
    bf16 = "unsupported"
    if bf16_supported:
        dtypes.append(torch.bfloat16)
        bf16 = "passed (bitwise gather, axis equality, quantized float32 oracle)"
    for dtype in dtypes:
        for s_size in args.s:
            progress("cuda_table_setup", dtype=str(dtype), S=s_size)
            current = make_tables(args.seed, args.roles, args.kv_heads, s_size, dtype)
            if dtype == torch.float16:
                tables[s_size] = current
            for kind in INPUT_KINDS:
                for config in CONFIGS:
                    records.append(correctness_case(args.seed, args.roles, args.kv_heads,
                                                     s_size, dtype, current, kind, config,
                                                     progress))
    return tables, records, bf16, bf16_supported


def tune(args, s_size, tables):
    cph = 4096 * args.head_dim // 4
    ids = torch.randint(24 * s_size, (args.roles * args.kv_heads * cph,), dtype=torch.int32,
                        generator=torch.Generator().manual_seed(args.seed)).cuda()
    out = {v: torch.empty((ids.numel(), 4), dtype=torch.float16, device="cuda") for v in VARIANTS}
    scores, selected = {}, {}
    for variant in VARIANTS:
        scores[variant] = {}
        for config in CONFIGS:  # Correctness already compiled every choice; time only now.
            fn = lambda v=variant, c=config: launch(v, ids, tables, out[v], cph, s_size, c)
            scores[variant][config[2]] = float(
                triton.testing.do_bench(fn, warmup=25, rep=50, quantiles=[0.5]))
        selected[variant] = min(CONFIGS, key=lambda c: scores[variant][c[2]])
    del ids, out
    torch.cuda.empty_cache()
    return selected, scores


def steady_warmup(fn, warmup_ms=25):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        fn()
    end.record()
    torch.cuda.synchronize()
    count = max(1, math.ceil(warmup_ms / max(start.elapsed_time(end) / 5, 1e-4)))
    for _ in range(count):
        fn()
    torch.cuda.synchronize()


def measure(args, s_size, t_kv, tables, selected):
    cph = t_kv * args.head_dim // 4
    chunks = args.roles * args.kv_heads * cph
    ids = torch.randint(24 * s_size, (chunks,), dtype=torch.int32,
                        generator=torch.Generator().manual_seed(args.seed)).cuda()
    out = {v: torch.empty((chunks, 4), dtype=torch.float16, device="cuda") for v in VARIANTS}
    fns = {v: (lambda v=v: launch(v, ids, tables, out[v], cph, s_size, selected[v]))
           for v in VARIANTS}
    for fn in fns.values():
        fn()
    torch.cuda.synchronize()
    rep, attempts = args.rep, []
    while True:
        rng, trials = random.Random(args.seed + s_size * 100000 + t_kv + rep), []
        for _ in range(args.outer_trials):
            order, trial = list(VARIANTS), {}
            rng.shuffle(order)
            for variant in order:
                cold = triton.testing.do_bench(fns[variant], warmup=args.warmup, rep=rep,
                                               quantiles=list(Q))
                steady_warmup(fns[variant], args.warmup)
                hot = triton.testing.do_bench_cudagraph(fns[variant], rep=rep, quantiles=list(Q))
                trial[variant] = {"cold": list(map(float, cold)), "hot": list(map(float, hot))}
            trial["order"] = order
            trials.append(trial)
        agg = {v: {mode: [statistics.median(t[v][mode][i] for t in trials) for i in range(3)]
                   for mode in ("cold", "hot")} for v in VARIANTS}
        stability = {v: (agg[v]["hot"][2] - agg[v]["hot"][0]) / agg[v]["hot"][1]
                     for v in ("J", "H")}
        attempts.append({"rep_ms": rep, "trials": trials, "aggregate": agg,
                         "hot_stability": stability})
        if max(stability.values()) <= 0.05 or rep >= 1600:
            break
        rep = min(rep * 2, 1600)
    output_bytes = chunks * 4 * 2
    row = {"S": s_size, "Tkv": t_kv, "roles": args.roles, "kv_heads": args.kv_heads,
           "head_dim": args.head_dim, "chunks": chunks, "output_bytes": output_bytes,
           "warmup_ms": args.warmup, "rep_ms": rep, "outer_trials": args.outer_trials,
           **{f"{v.lower()}_config": selected[v][2] for v in VARIANTS}}
    for variant in VARIANTS:
        for mode in ("cold", "hot"):
            p20, p50, p80 = agg[variant][mode]
            prefix = f"{variant.lower()}_{mode}"
            seconds = p50 / 1000
            row.update({f"{prefix}_p20_ms": p20, f"{prefix}_p50_ms": p50,
                        f"{prefix}_p80_ms": p80,
                        f"{prefix}_gchunks_s": chunks / seconds / 1e9,
                        f"{prefix}_output_gib_s": output_bytes / seconds / 2**30,
                        f"{prefix}_stability": (p80 - p20) / p50})
    row["jh_hot_speedup"] = row["j_hot_p50_ms"] / row["h_hot_p50_ms"]
    row["jh_cold_speedup"] = row["j_cold_p50_ms"] / row["h_cold_p50_ms"]
    row["stable"] = max(row["j_hot_stability"], row["h_hot_stability"]) <= 0.05
    del ids, out
    torch.cuda.empty_cache()
    return row, attempts


def correctness_summary(records, s_size):
    selected = [r for r in records if r["dtype"] == "float16" and r["S"] == s_size]
    result = {}
    for variant in ("F", "H"):
        for ref in ("J", "oracle"):
            key = f"{variant}_vs_{ref}"
            result[f"{variant.lower()}_max_abs_vs_{ref.lower()}"] = max(
                r["metrics"][key]["max_abs"] for r in selected)
            result[f"{variant.lower()}_relative_fro_vs_{ref.lower()}"] = max(
                r["metrics"][key]["relative_fro"] for r in selected)
    return result


def decide(rows):
    for row in rows:
        if not row["stable"]:
            row["decision"] = "UNSTABLE — DO NOT INTERPRET"
        elif row["Tkv"] == 4096:
            row["decision"] = ("KILL/RETARGET — >5% SHORT REGRESSION"
                               if row["h_hot_p50_ms"] > 1.05 * row["j_hot_p50_ms"]
                               else "SHORT-CONTEXT PASS")
        elif row["S"] == 192 and row["Tkv"] == 32768:
            speedup = row["jh_hot_speedup"]
            row["decision"] = "GO" if speedup >= 1.25 else (
                "OPTIMIZE ONCE" if speedup >= 1.10 else "KILL")
        else:
            row["decision"] = "DIAGNOSTIC"
    primary = next(r for r in rows if r["S"] == 192 and r["Tkv"] == 32768)
    if not primary["stable"]:
        return "UNSTABLE — DO NOT INTERPRET"
    if any(r["Tkv"] == 4096 and r["h_hot_p50_ms"] > 1.05 * r["j_hot_p50_ms"] for r in rows):
        return "KILL/RETARGET"
    return "GO" if primary["jh_hot_speedup"] >= 1.25 else (
        "OPTIMIZE ONCE" if primary["jh_hot_speedup"] >= 1.10 else "KILL")


def write_correctness_artifacts(output, records, bf16, bf16_supported):
    payload = {
        "schema": "vq-phase-a-correctness/v2",
        "status": "passed",
        "scientific_classification": "PASS",
        "mode": "correctness-only",
        "timing_executed": False,
        "tuning_executed": False,
        "cuda_graphs_executed": False,
        "bf16": bf16,
        "bf16_supported": bf16_supported,
        "tolerances": {"absolute": ABS_TOL, "relative_frobenius": REL_TOL},
        "coverage": {
            "S": list(SECONDARY_SIZES),
            "primary_units": PRIMARY_UNITS,
            "inputs": list(INPUT_KINDS),
            "configs": [config[2] for config in CONFIGS],
            "checks": ["finite", "J_bitwise_gather", "F_H_axis_bitwise_J",
                       "F_H_vs_independent_PyTorch_oracle"],
        },
        "records": records,
        "contract": contract_snapshot(bf16_supported=bf16_supported),
    }
    atomic_write_json(output / "correctness.json", payload)
    atomic_write_json(output / "correctness_manifest.json",
        {key: payload[key] for key in ("schema", "mode", "timing_executed", "tuning_executed",
                                       "cuda_graphs_executed", "bf16", "bf16_supported",
                                       "tolerances", "coverage", "contract")})
    atomic_write_text(output / "README.md",
        "# Phase A GPU correctness-only result\n\n"
        "This run executes only the exhaustive/random J/F/H correctness path. "
        "It performs no tuning, timing, CUDA graph, latency, or Phase A decision work.\n")


def write_artifacts(output, rows, raw, correctness, bf16, tuning, conclusion):
    with (output / "timings.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output / "correctness.json").write_text(json.dumps(
        {"status": "passed", "bf16": bf16, "records": correctness}, indent=2) + "\n")
    (output / "tuning.json").write_text(json.dumps(tuning, indent=2) + "\n")
    (output / "trial_timings.json").write_text(json.dumps(raw, indent=2) + "\n")
    lines = ["# Phase A Hurwitz decode baseline", "", f"Conclusion: **{conclusion}**.", "",
             "Independent paper-spec synthetic reconstruction; scalar-first `(w,x,y,z)`; `id=p*S+s`.",
             "Correctness passed before timing on exhaustive `(p,s)` and random inputs. J matched an independent "
             "PyTorch gather bitwise; F/H axes matched J bitwise; fp16 tolerances and finite-value checks passed.",
             f"bf16 correctness-only coverage: {bf16}.", "",
             "| S | Tkv | J hot us | F hot us | H hot us | J/H hot | J/H cold | H stability | "
             "H max abs | H rel Fro | configs J/F/H | Decision |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|:---|"]
    for r in rows:
        lines.append(f"| {r['S']} | {r['Tkv']} | {r['j_hot_p50_ms']*1000:.3f} | "
                     f"{r['f_hot_p50_ms']*1000:.3f} | {r['h_hot_p50_ms']*1000:.3f} | "
                     f"{r['jh_hot_speedup']:.3f} | {r['jh_cold_speedup']:.3f} | "
                     f"{r['h_hot_stability']:.4f} | {r['h_max_abs']:.3g} | "
                     f"{r['h_relative_fro']:.3g} | {r['j_config']}/{r['f_config']}/{r['h_config']} | "
                     f"{r['decision']} |")
    lines += ["", "Primary gate: steady-state p50 at `S=192,Tkv=32768`; GO >=1.25x, "
              "OPTIMIZE ONCE >=1.10x, otherwise KILL. H >5% slower at Tkv=4096 forces KILL/RETARGET.",
              "Full p20/p50/p80, Gchunks/s, output GiB/s, configurations, and decisions are in `timings.csv`."]
    (output / "README.md").write_text("\n".join(lines) + "\n")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for s_size in sorted({r["S"] for r in rows}):
        chosen = sorted((r for r in rows if r["S"] == s_size), key=lambda r: r["Tkv"])
        ax.plot([r["Tkv"] for r in chosen], [r["jh_hot_speedup"] for r in chosen], "o-", label=f"S={s_size}")
    ax.axhline(1.10, color="#d97706", ls="--", label="1.10")
    ax.axhline(1.25, color="#15803d", ls="--", label="1.25")
    ax.set(xscale="log", xlabel="Tkv", ylabel="Steady-state J/H speedup", title="Phase A Hurwitz decode")
    ax.set_xticks([4096, 16384, 32768], ["4K", "16K", "32K"])
    ax.grid(alpha=.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "jh_speedup.png", dpi=180)
    plt.close(fig)


def git(*args):
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def read_execution_boundary(output):
    try:
        return read_json(output / STATE_FILE).get("boundary", "unknown_infrastructure")
    except (OSError, ValueError, json.JSONDecodeError):
        return "unknown_infrastructure"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--s", type=int, nargs="+", default=[96, 192])
    p.add_argument("--t-kv", type=int, nargs="+", default=[4096, 16384, 32768])
    p.add_argument("--roles", type=int, default=2)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--dtype", choices=["fp16"], default="fp16")
    p.add_argument("--cache-modes", nargs="+", default=["hot", "cold"])
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--rep", type=int, default=200)
    p.add_argument("--outer-trials", type=int, default=5)
    p.add_argument("--output", type=Path, default=Path("results/2026-08-20-hurwitz-decode-baseline"))
    p.add_argument("--correctness-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    validate_correctness_arguments(
        CorrectnessArguments(
            correctness_only=args.correctness_only,
            seed=args.seed,
            secondary_sizes=tuple(args.s),
            roles=args.roles,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
        )
    )
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_sha256 = os.environ.get("PHASE_A_MANIFEST_SHA256", "")
    if re.fullmatch(r"[0-9a-f]{64}", manifest_sha256) is None:
        record_boundary(args.output, "payload_before_cuda", error="launch manifest digest missing")
        raise RuntimeError("PHASE_A_MANIFEST_SHA256 must bind the immutable launch manifest")
    record_boundary(
        args.output,
        "payload_before_cuda",
        launch_manifest_sha256=manifest_sha256,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run only through the staged Slurm entry point")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    metadata = {"schema": "vq-phase-a-run-metadata/v2",
                "status": "running", "git_commit": git("rev-parse", "HEAD"),
                "git_branch": git("branch", "--show-current"), "cwd": os.getcwd(),
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "source_repo": os.environ.get("SOURCE_REPO"),
                "staged_repo": os.environ.get("RESEARCH_REPRO_STAGED_DIR"),
                "result_root": str(args.output),
                "torch": torch.__version__, "triton": triton.__version__,
                "cuda_runtime": torch.version.cuda, "device": torch.cuda.get_device_name(0),
                "compute_capability": torch.cuda.get_device_capability(0),
                "mode": "correctness-only" if args.correctness_only else "timing",
                "correctness_only": True,
                "launch_manifest_sha256": manifest_sha256,
                "timing_executed": False, "tuning_executed": False,
                "cuda_graphs_executed": False,
                "run_complete": False,
                "assumptions": {"quaternion": "scalar-first (w,x,y,z)", "id": "p*S+s"}}
    atomic_write_json(args.output / "run_metadata.json", metadata)
    print("correctness: starting", flush=True)
    progress = lambda boundary, **details: record_boundary(
        args.output,
        boundary,
        launch_manifest_sha256=manifest_sha256,
        **details,
    )
    try:
        tables, correctness, bf16, bf16_supported = run_correctness(args, progress)
    except BaseException as exc:
        scientific_failure = isinstance(exc, ScientificComparisonFailure)
        failure = {
            "schema": "vq-phase-a-correctness/v2",
            "status": "failed",
            "mode": metadata["mode"],
            "scientific_classification": "FAIL" if scientific_failure else "NO RESULT",
            "failure_kind": "numerical_comparison" if scientific_failure else "infrastructure",
            "failure_phase": "inside_comparison" if scientific_failure else
                             read_execution_boundary(args.output),
            "failed_check": exc.check if scientific_failure else None,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        atomic_write_json(args.output / "correctness.json", failure)
        metadata.update(
            status="failed",
            correctness=failure["scientific_classification"],
            failure_phase=failure["failure_phase"],
            error=failure,
        )
        atomic_write_json(args.output / "run_metadata.json", metadata)
        raise
    print(f"correctness: passed; bf16: {bf16}", flush=True)
    write_correctness_artifacts(args.output, correctness, bf16, bf16_supported)
    progress("comparison_output_written", scientific_classification="PASS")
    metadata.update(status="comparison-passed", correctness="PASS")
    atomic_write_json(args.output / "run_metadata.json", metadata)
    print("correctness-only comparisons: PASS; no tuning or timing executed", flush=True)
    return
    selected, tuning = {}, {"configs": [c[2] for c in CONFIGS], "selection_shape_tkv": 4096, "by_s": {}}
    metadata["tuning_executed"] = True
    for s_size in args.s:
        selected[s_size], scores = tune(args, s_size, tables[s_size])
        tuning["by_s"][str(s_size)] = {"scores_ms": scores,
                                      "selected": {v: selected[s_size][v][2] for v in VARIANTS}}
    rows, raw = [], {}
    summaries = {s: correctness_summary(correctness, s) for s in args.s}
    metadata["timing_executed"] = True
    metadata["cuda_graphs_executed"] = True
    for s_size in args.s:
        for t_kv in args.t_kv:
            print(f"timing: S={s_size}, Tkv={t_kv}", flush=True)
            row, attempts = measure(args, s_size, t_kv, tables[s_size], selected[s_size])
            row.update(summaries[s_size])
            row["h_max_abs"] = max(row["h_max_abs_vs_j"], row["h_max_abs_vs_oracle"])
            row["h_relative_fro"] = max(row["h_relative_fro_vs_j"], row["h_relative_fro_vs_oracle"])
            rows.append(row)
            raw[f"S{s_size}-T{t_kv}"] = attempts
    conclusion = decide(rows)
    write_artifacts(args.output, rows, raw, correctness, bf16, tuning, conclusion)
    metadata.update(status="passed", conclusion=conclusion)
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"conclusion: {conclusion}", flush=True)


if __name__ == "__main__":
    main()
