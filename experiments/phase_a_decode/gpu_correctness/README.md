# Phase A clean GPU correctness rerun

Current state: preparation only. Nothing in this directory selects a live resource or
authorizes a submission. Do not reuse job `1579631`, its stage, result root, ledger, or
event source.

The rerun stays correctness-only: `S={96,192}`, two roles, eight KV heads, head dimension
128, exhaustive and seeded-random inputs, and launch configurations `b256-w4` and
`b512-w8`. Each supported dtype has eight records. Exhaustive records contain 36,864
chunks for `S=96` and 73,728 for `S=192`; random records contain 65,584 chunks. Fp16
keeps `max_abs <= 4e-3` and `relative_fro <= 1e-3`; bf16 keeps the previously declared
quantized-oracle checks and records the direct-float32 metric without adding a threshold.
Timing, tuning, CUDA graphs, plots, speedups, and GO/OPTIMIZE/KILL decisions are rejected.

## Preparation gates

Run these only after a separate launch authorization. Use new canonical paths under
`/data/scratch-fast/kwen1/compute-native-vq`; never repair or reuse an old stage.

1. Record the final clean commit `R` and tree `T`.
2. Create a new standalone source clone from the clean research worktree:

   ```bash
   python3 experiments/phase_a_decode/gpu_correctness/prepare_clean_rerun.py \
     clone-source \
     --source "$SOURCE_WORKTREE" \
     --destination "$STANDALONE_SOURCE" \
     --commit "$R"
   ```

   This uses `git clone --no-local --no-hardlinks`, detaches at `R`, rejects alternates
   and shared object inodes, and requires a self-owned `.git` directory plus a full
   `git fsck` pass.

3. From the detached standalone clone, run the CPU preflight with CUDA hidden:

   ```bash
   "$PYTHON" experiments/phase_a_decode/gpu_correctness/prepare_clean_rerun.py \
     cpu-preflight \
     --repo "$STANDALONE_SOURCE" \
     --output "$RESULT_ROOT/cpu_preflight.json" \
     --commit "$R"
   ```

   It records exact commands, outputs, return codes, package versions, protocol values,
   executable hashes, commit, tree, and clean detached state. Any failure stops before
   staging.

4. Immediately before preparation, the sole future scheduler owner must capture current
   associations and eligible Torralba-only resources. Write the raw evidence into the new
   result root and a `vq-phase-a-scheduler-selection/v1` JSON contract beside it. The
   contract must contain the evidence filename/hash, timestamp, exact account/QoS/partition,
   selection rationale, one node/task/GPU, at most four CPUs, 16 GiB, at most 15 minutes,
   advertised GPU names, positive minimum VRAM, and required CUDA/Triton/one-device checks.
   It must also declare bf16 coverage as required whenever the selected GPU supports it.
   `freeze-stage` rejects evidence older than 15 minutes. It has no default scheduler tuple.

   ```json
   {
     "schema": "vq-phase-a-scheduler-selection/v1",
     "captured_at_utc": "<timezone-aware timestamp>",
     "account": "<live eligible account>",
     "qos": "<live eligible qos>",
     "partition": "<live eligible Torralba partition>",
     "torralba_only": true,
     "selection_rationale": "<why the advertised GPU is adequate>",
     "resources": {
       "nodes": 1, "tasks": 1, "cpus": 4, "gpus": 1,
       "memory_gib": 16, "time_minutes": 15
     },
     "adequacy": {
       "advertised_gpu_names": ["<exact runtime device name>"],
       "minimum_vram_bytes": 8589934592,
       "cuda_required": true,
       "triton_required": true,
       "exactly_one_visible_gpu": true,
       "bf16_policy": "required-when-supported"
     },
     "selection_evidence": {
       "file": "scheduler_preflight.txt",
       "sha256": "<sha256 of the raw evidence file>"
     }
   }
   ```

5. Invoke the research staging helper from the standalone clone. Its `rsync -a` now copies
   a real stage-owned `.git` directory rather than an external worktree pointer:

   ```bash
   (
     cd "$STANDALONE_SOURCE"
     /afs/csail.mit.edu/u/k/kwen1/.codex/skills/research-reproducibility/scripts/stage_and_run.sh \
       --repo-root "$STANDALONE_SOURCE" \
       --staging-parent /data/scratch-fast/kwen1/compute-native-vq/staging \
       -- "$PYTHON" experiments/phase_a_decode/gpu_correctness/prepare_clean_rerun.py \
         freeze-stage \
         --source "$STANDALONE_SOURCE" \
         --result "$RESULT_ROOT" \
         --cpu-preflight "$RESULT_ROOT/cpu_preflight.json" \
         --scheduler-contract "$RESULT_ROOT/scheduler_selection.json" \
         --commit "$R" \
         --tree "$T" \
         --python "$PYTHON"
   )
   ```

   The staged command verifies source/stage identity, clean status, detached HEAD/tree,
   stage-owned Git common/object paths, no alternates, object ownership/link counts, full
   `fsck`, tracked count/content, exact metadata, and the declared untracked set. It then
   makes the entire stage read-only and binds its recursive path/type/mode/size/SHA-256
   inventory into `launch_manifest.json`.

## Prepared launch boundary

A successful preparation creates these files in the fresh result root:

- `launch_manifest.json` and `launch_manifest.sha256`: immutable commit/tree, stage
  inventory, metadata/preflight/config hashes, selected scheduler tuple, exact benchmark
  argv, result schema, and correctness/no-timing contract;
- `submission_request.json`: exact one-shot `sbatch` argv and manifest-bound submission
  environment;
- `attempt_ledger.jsonl`: an fsynced hash chain beginning with zero submissions and no
  accepted job ID.

The manifest and request are mode `0444`. The ledger is only extended through the
exclusive-create launch lock and append-only writer in `submit_from_stage.py`. A failed,
ambiguous, or accepted invocation permanently consumes the attempt; a second invocation
is rejected. The submitter sets the added variables in the `sbatch` process environment
and omits `--export`, so Slurm's safe default `ALL` export applies; `--export=ALL,...`,
`--export=NONE`, and `--export=NIL` are not used.

Submission remains a separate launch-time action. If freshly authorized, run the prepared
submit command once from the frozen stage. After an accepted numeric ID, register exactly
one new fixed-host/fixed-job terminal event source. No conversational polling or retry is
allowed.

At runtime, the manifest digest is written into the first artifact, environment validation,
run metadata, and final manifest. Only an identified finite, bitwise, or declared tolerance
failure is scientific `FAIL`. Environment, allocation, compilation, launch, serialization,
malformed/incomplete evidence, and post-write failures are `NO RESULT`; a post-write failure
preserves the completed comparison status separately.
