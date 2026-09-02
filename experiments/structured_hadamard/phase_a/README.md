# Phase A structured-Hadamard slice

This directory is the smallest code-and-validation slice for the first
Llama-2-7B FFN `down_proj` input. It has not produced a completed GPU benchmark,
driver output, or measurement rows. The preserved notebook records earlier
scheduler allocations, CUDA visibility and basic-operation checks, and an
untimed A4 correctness attempt that failed during Triton compilation; the final
r2 owner attempt failed before SSH or Slurm. All emitted preflight rows remain
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

Only `I` and `Hfull` are valid Phase A measurement transforms. `H32` and
`H128` remain CPU/reference specifications for later phases and the schema
rejects measurement rows labeled with either one.

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
identities. The latter is explicitly sequential transform-then-quantize
timing, requires the future owner to provide the frozen quantizer callback,
and records `fusion="none"`. It makes no fused-kernel claim. This slice
deliberately does not build an INT4 GEMM or claim W4A4/e2e evidence.

`activation_quantizer.py` now supplies that one narrow callback: one Triton
launch implementing deterministic round-to-nearest-even, dynamic per-row,
symmetric signed A4 (`[-7, 7]`) and writing a dequantized fp16 tensor plus fp32
row scales. It is compatible with a later W4A4 linear boundary, but it does
not pack nibbles, convert weights, execute a GEMM, or fuse with `Hfull`.

## Record contract

`schema.py` is the authoritative `rot-site-v1.phase-a.2` JSONL validator. It
requires exact nested fields for code, model, quantization, site/transform,
workload, hardware, metrics, timing, execution state, and artifact paths. It
rejects unknown/missing fields, duplicate JSON keys, duplicate logical rows,
non-finite values, wrong JSON types for integral fields,
dimension/factorization mismatches, wrong `U_172` digests, and identity rows
that claim transform work. Planned rows must have every metric set to null,
`scheduler_clearance=false`, and `scientific_evidence=false`.

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
python3 -m unittest experiments.structured_hadamard.phase_a.tests.test_execute -v
python3 -m unittest experiments.structured_hadamard.phase_a.tests.test_stage_repository -v
python3 -m experiments.structured_hadamard.phase_a.oracle \
  --seed 0 --token-rows 2 --weight-rows 3
python3 -m experiments.structured_hadamard.phase_a.preflight \
  --scheduler-clearance=false --format=jsonl \
  --result-jsonl /data/scratch-fast/kwen1/structured-hadamard/phase-a/results/phase-a.jsonl
```

No `make` is needed because this slice changes only Python and documentation.
The tests use `unittest` so the clean pinned checkout needs no dependency
installation; they remain pytest-discoverable when pytest is available.

## Real-model W4A4 preparation

`real_model/` contains the bounded helper for a genuine packed W4A4 path. It
binds only QuaRot's unchanged signed-int4 quantization, CUTLASS int4-by-int4
GEMM, and int32 dequantization sources. It does not bind KV-cache/FlashInfer
code, implement a replacement kernel, or use an int8-container surrogate.
Native code is limited to verified SM80/SM86 SASS plus compute-80 PTX; SM70 is
unsupported, and any newer device requires a later PTX-JIT compatibility
check.

The real-model loader verifies and loads the exact cached Llama-3 checkpoint
through standard Transformers eager attention, derives GQA head dimension as
`hidden_size / num_attention_heads = 128`, and rejects uninitialized meta
tensors or a state-dict/index mismatch. It replaces only explicitly selected
`model.layers[*].mlp.down_proj` modules. The selected real weight is folded by
the same exact QuaRot full-Hadamard factorization applied online before packed
W4A4 quantization, GEMM, and row/column dequantization.

The earlier login-safe build did not produce an artifact; its preserved
failure evidence remains under `data/rot-phasea-real-model-baseline-r1/`.
Acceptance of any later CPU-allocation artifact is recorded in its separate
task report and must not be inferred from the presence of these helpers.

The decisive real-model driver is `real_model/benchmark.py`. It compares one
FP16 model with two genuine packed-W4A4 variants. Both W4A4 variants replace
all 224 Transformer q/k/v/o/gate/up/down projection linears; embeddings,
normalization, and the LM head remain floating point. The variants differ only
at the 32 FFN `down_proj` inputs: one uses the host-alias identity and the other
uses the exact online full Hadamard with matching folded weights. Every such
transform and packed GEMM is sequential and records `fusion="none"`.

`real_model/preflight_remote_audit.py` is the only remote Python preflight. It
uses the standard library and `importlib.metadata`, never imports a scientific
package merely to report a version, and consumes one exact mode-0600 clearance
through ordinary argv. `real_model/gpu_owner.py` owns the sole `salloc` and
`srun --pty`; `real_model/local_event_owner.py` reuses the one existing SSH
ControlMaster and emits the one tmux terminal signal.

## Future experimental boundary

Only a later, explicitly cleared GPU owner may activate profiling. Before any
experimental command, that owner must use `stage_repository.py` to create a
complete pinned repository—including an ordinary independent `.git`
directory, the exact source `HEAD`, dirty tracked content, and all untracked
inputs, including Git-ignored repo-local configs/scripts/inputs—and then run
entirely inside the staged copy. In addition to
`REPRODUCIBILITY_METADATA.json`, the v3 stage now materializes
`REPRODUCIBILITY_MANIFEST.json`, containing every copied input identity; the
metadata binds that file and its canonical entry digest.

The only default working-tree exclusions are the recursively excluded root
trees `staging`, `out`, `outputs`, `eval_outputs`, `slurm_outputs`, and
`wandb`; cache/virtualenv directory names `.cache`, `__pycache__`,
`.pytest_cache`, `.mypy_cache`, `.ruff_cache`, `.venv`, and `venv` are
excluded at any depth. Tracked files always override these exclusions. Source
`.git` metadata is handled separately by the independent clone. The helper
supports linked-worktree sources and refuses existing/in-repository
destinations, unmerged indexes, submodules it cannot prove independent,
special included input files, included paths with symlink ancestors, symlinks
that are absolute, escape the source, or traverse content absent from the
copied closure, reserved control-file collisions, external Git object
alternates, source changes during copying, or any failed post-copy verification.

The bounded staging assertion used during preparation is:

```bash
python3 -m experiments.structured_hadamard.phase_a.stage_repository \
  --source /path/to/source/triton \
  --destination /data/scratch-fast/kwen1/structured-hadamard/staging/<timestamp>-<commit>-code \
  -- git rev-parse --verify 'HEAD^{commit}'
```

That command only asserts the staged Git identity; it is not a benchmark or
scientific workload. The existing `preflight.py --execute` refusal remains
unchanged and cannot activate profiling.

`execute.py` is the separate owner-only boundary. It accepts only an absolute,
non-symlink clearance JSON owned by the executing uid at mode 0600 and located
outside the stage. The exact `phase-a-scheduler-clearance-v1` keys are:

```json
{
  "schema_version": "phase-a-scheduler-clearance-v1",
  "scheduler_clearance": true,
  "clearance_id": "OWNER-SUPPLIED-ID",
  "owner": "OWNER-NAME",
  "owner_uid": 28131,
  "stage_root": "/absolute/complete/stage",
  "source_commit": "40-HEX-STAGED-HEAD",
  "driver_commit": "40-HEX-STAGED-HEAD",
  "transform_commit": "40-HEX-STAGED-HEAD",
  "stage_manifest_sha256": "64-HEX-MANIFEST-DIGEST",
  "device": "cuda:0",
  "output_directory": "/absolute/new/durable/output-directory",
  "model_revision": "40-HEX-MODEL-REVISION",
  "site_layer": 0,
  "quant": {
    "w_bits": 4,
    "a_bits": 4,
    "w_group_size": "128",
    "a_group_size": "per-row",
    "w_symmetric": true,
    "a_symmetric": true,
    "scale_granularity": "per-group-W4;dynamic-per-row-A4",
    "clip": "none",
    "calibration_dataset": "DATASET-NAME@40-HEX-REVISION",
    "calibration_seed": 0,
    "calibration_rows": 8192
  },
  "timing": {"warmup_ms": 25, "repetition_ms": 200, "outer_trials": 5},
  "clock_policy": "OWNER-OBSERVED-POLICY"
}
```

Quantization and timing counts must be non-boolean JSON integers. Activation
metadata must select symmetric dynamic-per-row A4 through `a_group_size`,
`a_symmetric`, and exactly one `dynamic-per-row-A4` granularity component.
Every commit field must equal the clean staged `HEAD`; the manifest digest
must equal the stage metadata. The model and calibration revisions must be
immutable 40-hex pins. The device must include one explicit CUDA index, and
the canonical output directory must not exist and must be outside the stage.
The driver re-hashes the entire stage before any Torch/Triton import or input
creation and rejects source/untracked changes, incomplete Git metadata,
reference-only transform measurements, unresolved pins, non-finite/partial
samples, and output overwrite.

Only after those checks does it run the existing full-width oracle, create the
fixed-seed `[1,11008]` fp16 tensor on the clearance-selected device, check the
GPU `I`/`Hfull` and A4 callback against the CPU specifications, and profile the
four accepted rows through `profiler.profile_transform`. A later cleared owner
runs, from the stage root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m experiments.structured_hadamard.phase_a.execute \
  --scheduler-clearance-file /absolute/owner-clearance.json
```

The new durable output directory contains validated
`rot-site-v1.phase-a.2` `phase-a.jsonl`, `phase-a.raw-samples.jsonl`, and a
digest-bearing `phase-a.execution.json`. The main JSONL file is committed
last, all files use atomic writes, and any caught failure removes only files
created by that invocation. Every record remains visibly synthetic,
non-model, non-PPL, and `scientific_evidence=false`. The execution driver has
no resource-acquisition, remote-access, job-control, retry, or lifecycle
ownership behavior; those decisions remain entirely outside this repository
boundary.
