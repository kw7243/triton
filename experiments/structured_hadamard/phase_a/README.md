# Phase A structured-Hadamard slice

This directory is the smallest code-and-validation slice for the first
Llama-2-7B FFN `down_proj` input. It has not run a GPU benchmark, model,
quantizer, CUDA command, or scheduler job. All emitted preflight rows are
`UNEXECUTED` and are not scientific evidence.

## Fixed semantics

Rows are transformed on the right and PyTorch stored weights are folded in the
same orientation:

```text
Z = X R
A_folded = A R
Z A_folded^T = X A^T
```

The reference menu is:

- `I`: a host alias; no transform launch and no copy;
- `H32`: normalized natural-order Sylvester transforms over consecutive
  32-channel blocks;
- `H128`: the analogous consecutive 128-channel blocks;
- `Hfull`: QuaRot-compatible `11008=172x64`, with the exact pinned `U_172`, a
  natural-order 64-point FHT, and one `1/sqrt(11008)` normalization.

There are no random signs, permutations, padding/truncation, or learned stages.
The canonical pinned matrix identity is
`sha256:int8-row-major:378ef12c7cc31f3ea558e1c66b9552a83128f8f732b0bc094b864c8e120d84bd`.
The Apache-2.0 attribution is in `THIRD_PARTY_NOTICES.md`.

`reference.py` is dependency-free and independent of the Triton kernels.
`triton_transform.py` lazily defines a two-launch `Hfull` path: an on-chip
64-point FHT with an fp32 intermediate (preventing unnormalized fp16 overflow)
and a masked/tiled `U_172` reduction. Padding the reduction tile to 256 is an
implementation mask only; it never pads or truncates the 11008-wide operator.
The `I` branch returns before importing Torch or Triton.

`profiler.py` keeps `transform-only` and `transform+quantize` as distinct timing
identities. The latter requires the future owner to provide the frozen
quantizer callback; this slice deliberately does not build an INT4 GEMM or
claim W4A4/e2e evidence.

## Record contract

`schema.py` is the authoritative `rot-site-v1.phase-a.1` JSONL validator. It
requires exact nested fields for code, model, quantization, site/transform,
workload, hardware, metrics, timing, execution state, and artifact paths. It
rejects unknown/missing fields, duplicate JSON keys, duplicate logical rows,
non-finite values, dimension/factorization mismatches, wrong `U_172` digests,
and identity rows that claim transform work. Planned rows must have every
metric set to null, `scheduler_clearance=false`, and
`scientific_evidence=false`.

Print the exact four-row Phase A matrix (`I` and `Hfull`, each with the two
separate timing identities) without touching CUDA:

```bash
python3 -m experiments.structured_hadamard.phase_a.preflight \
  --scheduler-clearance=false \
  --result-jsonl /data/scratch-fast/kwen1/structured-hadamard/phase-a/results/phase-a.jsonl
```

The preparation CLI accepts no `true` clearance value. Adding `--execute`
returns status 2 before code discovery or any GPU import:

```bash
python3 -m experiments.structured_hadamard.phase_a.preflight \
  --scheduler-clearance=false --execute
```

## Local validation

These commands are CPU/static only and must be run from the repository root:

```bash
python3 -m compileall -q experiments/structured_hadamard/phase_a
python3 -m unittest discover -s experiments/structured_hadamard/phase_a/tests -v
python3 -m experiments.structured_hadamard.phase_a.oracle \
  --seed 0 --token-rows 2 --weight-rows 3
python3 -m experiments.structured_hadamard.phase_a.preflight \
  --scheduler-clearance=false --format=jsonl \
  --result-jsonl /data/scratch-fast/kwen1/structured-hadamard/phase-a/results/phase-a.jsonl
```

No `make` is needed because this slice changes only Python and documentation.
The tests use `unittest` so the clean pinned checkout needs no dependency
installation; they remain pytest-discoverable when pytest is available.

## Future experimental boundary

Only a later, explicitly cleared GPU owner may activate profiling. Before any
experimental command, that owner must copy the full repository—including
`.git`, this branch, configs, and uncommitted inputs—then run entirely inside
the staged copy:

```bash
cd /path/to/source/triton
/home/ubuntu/.codex/skills/research-reproducibility/scripts/stage_and_run.sh -- \
  python3 -m experiments.structured_hadamard.phase_a.preflight \
    --scheduler-clearance=false --execute \
    --result-jsonl /data/scratch-fast/kwen1/structured-hadamard/phase-a/results/phase-a.jsonl
```

That exact command intentionally refuses today. The future owner must first
land a new provenance commit containing an administrator-authenticated
clearance record and an owner-only execution driver which consumes these
validated rows and calls `profiler.profile_transform`. The transform semantics,
matrix digest, timing identities, and JSONL contract do not need redesign.
Submission and result paths must remain under `/data/scratch-fast/kwen1`, and
submission must occur from the completed staged repository rather than this
source worktree.
