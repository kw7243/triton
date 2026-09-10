# Structured Rotations v2 research notebook

## 2026-09-10 — Gate A execution intake and A2 freeze

- Branch: `fm/structured-rotations-v2-gate-a-execution-r1` from exact base
  `f893845b9b91599ebd3b7a9c7f28164f39c7ed94`. Delivery is local-only; no push,
  PR, merge, rebase, or default-branch change is authorized.
- Forward authority is
  `/home/ubuntu/firstmate/data/research-plans/idea7_structured_rotations_experimental_plan_v2.md`.
  This lane owns Experiment A2 only. It excludes quality experiments, Gate B,
  representative bridges, the rejected per-layer selector, and broad kernel work.
- The preserved Phase A/B/C worktrees were checked read-only. Their clean tips
  and trees exactly match the direct-handoff report: A `3bc279eca8` /
  `8b34fdb5df`, B `f7cd759307` / `43e1d8b804`, and C `8c8853f29a` /
  `387558e8f0`. A's 22, B's 29, and C's 21 retained artifact entries rehashed
  successfully after using each manifest's own root. No old stage, branch, or
  selector artifact was changed or imported as a v2 result.
- The old evidence establishes only that full rotation had a large causal effect
  in a poor Llama-3-8B W4A4 pipeline, that fixed block transforms were cheaper,
  and that the old selector lost to the fixed frontier. Its isolated transform
  timings and absolute PPL values are not v2 Gate A completion evidence.
- The completed A1 quality lane was consumed read-only at clean local tip
  `3377b4ed346c3b363dbd285ade8d867bd8cedeb2`. It froze the permitted Qwen3-8B
  substitution, revision `b968826d9c46dd6066d109eabc6255188de91218`, because
  Llama-3.1-8B access was unavailable. A1 found pooled output error `0.030639`
  for full Hadamard, `0.096933` for PeRQ-32, and `0.047932` for PeRQ-128.
  Decision A therefore still needs native A2 cost evidence; Gate B is not open.
- The actual frozen FFN-down tensor contract is hidden width 4096, intermediate
  width 12288, down output width 4096, down weight `[4096,12288]`, packed down
  weight `[4096,6144]`, and activation row counts 2048, 1, and 8 for batch-1
  2K prefill, batch-1 2K-context decode, and batch-8 2K-context decode.
- The preserved native backend is QuaRot's CUTLASS signed-int4 tensor-core GEMM
  with int32 accumulation, artifact SHA-256
  `10a961e8855d7349708412aa8c21a38667fb75c94c372180dd6f992913e0a8e0`.
  The A1 unsigned activation alphabet is mapped exactly by packing `q_u-8` and
  adding `(8-zp)*sum(q_w)` before dequantization. The integer identity has a
  dependency-free regression test and must pass against the native consumer
  before any timing row is accepted.
- PeRQ is timed for the complete layer-0 segment because A1 retained its exact
  frozen permutation. It is not timed end to end because A1 did not freeze
  permutations for all 36 layers, and A2 cannot create them by running another
  calibration experiment. End-to-end comparisons are identity, optimized exact
  full Hadamard, local-32, and local-128 with all 36 FFN down projections using
  the same native packed consumer and common unfused boundaries.
- The complete segment includes BF16 gate/up production, SiLU and multiply,
  transform, dynamic scale/zero point, physical INT4 packing, native packed
  consumer, exact zero-point correction, and BF16 output. Transform-only timing
  and the prior Torch full implementation are supporting diagnostics only.
- Exact pre-outcome choices are in
  `experiments/structured_rotations_v2/gate_a/execution_contract.json`. The
  research-reproducibility helper will create one fresh complete stage directly
  under `/data/scratch-fast/kwen1/structured-rotations-v2/staging/`. The native
  artifact and A1 layer-0 cache are copied into the remote source before staging,
  then included in a complete stage file manifest. No experiment may start until
  that stage and its ordinary independent Git metadata verify.
- Retained outputs are disjoint from the quality lane under
  `/data/vision/torralba/u/kwen1/structured-rotations-v2/gate-a-execution-r1/`.
  Environments, model cache, compiler cache, transfer bundle, and temporary
  output remain under `/data/scratch-fast/kwen1`.
- Local CPU/static validation before staging: `compileall`, six focused
  `unittest` cases, and `git diff --check` pass. `make` was not run because the
  changes are Python and research documentation only. No experimental command
  ran on the Firstmate VM.

Exactly one next experiment: the frozen A2 job described above, after a fresh
Slurm inventory, ownership check, and successful `sbatch --test-only`.

## 2026-09-10 — A2 job 1824515 stopped before measurement

### Current state

- Decision A is not reached. Gate B remains closed.
- The only A2 submission, job `1824515`, failed before CUDA, model, native
  backend, tensor-contract, or timing validation. No segment or end-to-end
  latency row exists, and no retry or requeue was submitted.
- The exact failure is `ModuleNotFoundError: No module named 'pyarrow'`. The
  selected `causal_forcing` environment has Python 3.10.20, numpy 1.24.4,
  torch 2.8.0+cu128, transformers 5.12.1, tokenizers 0.22.2, and Triton 3.4.0,
  but no PyArrow package.

### Source and stage

- Pre-run source commit: `edd1bf833e6e50cb458684e077673203766ef4ea`.
- Source tree: `b7b75e7c944408c8fe129a20d26c1bcaa13a26e7`.
- Transfer bundle SHA-256: `48b5d58695a31822b7779440362f42bda3fd26c42ade947ed29ea86a0db6b6c0`.
- Reproducibility helper SHA-256:
  `44e5dc6f1a958b1f4b32e8dceeb49814885ad8dcca59716090ca87b1033731fa`.
- Full stage:
  `/data/scratch-fast/kwen1/structured-rotations-v2/staging/20260910_191426-dff84f-edd1bf833-code`.
- The stage has an ordinary `.git` directory, passed
  `git fsck --connectivity-only`, and resolved to the source commit and tree.
- `REPRODUCIBILITY_METADATA.json` SHA-256:
  `91e21f888951b91b92cfae2589a5a91c4717be1c0e32e4c4569709d059be2cff`.
- `STAGE_FILE_MANIFEST.json` has 1,750 entries, reverified byte for byte, and
  has SHA-256
  `c9b32c4a2b6ad1273804cf5039544621f41c25eb732a78c06a3576001ca2490a`.
- Execution contract SHA-256:
  `a46e11ebbe9daef59b5eaa7e472dcc57e97d2bb5e20acbc9d94c15dbe21f8deb`.
- Native extension SHA-256:
  `10a961e8855d7349708412aa8c21a38667fb75c94c372180dd6f992913e0a8e0`.
- Frozen A1 layer-0 cache SHA-256:
  `ac76a0aee1260dd9947c1cde11a4d54535d859fa277ce61c9402cfcb7e23b7d7`.

### Allocation and commands

- The live account, QoS, partition, and accelerator inventory was retained at
  `/data/vision/torralba/u/kwen1/structured-rotations-v2/gate-a-execution-r1/provenance/preflight-20260910T231852Z/`.
- The selected tuple was account `vision-torralba-urops-meng`, QoS
  `vision-torralba-interactive`, partition `vision-torralba-rtx3090`, and one
  `gpu:rtx_3090`. This is the smallest native-compatible Torralba allocation:
  the extension contains SM80/SM86 code, the RTX 3090 is SM86, Torralba V100s
  are incompatible, and H100/H200 do not match the compiled artifact.
- `sbatch --test-only` accepted the exact tuple and printed test-only ID
  `1824441`. Immediately before submission, matching queue, accounting, and
  retained-run checks were empty. An atomic single-submission latch was then
  created under scratch.
- Stage creation command:

  ```bash
  /data/scratch-fast/kwen1/structured-rotations-v2/gate-a-execution-r1/tools/stage_and_run.sh --repo-root /data/scratch-fast/kwen1/structured-rotations-v2/gate-a-execution-r1/source/edd1bf833e6e50cb458684e077673203766ef4ea-repo --staging-parent /data/scratch-fast/kwen1/structured-rotations-v2/staging --stage-only
  ```

- Manifest verification command:

  ```bash
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/data/scratch-fast/kwen1/structured-rotations-v2/staging/20260910_191426-dff84f-edd1bf833-code /data/scratch-fast/kwen1/micromamba/root/envs/causal_forcing/bin/python -m experiments.structured_rotations_v2.gate_a.stage_manifest --root /data/scratch-fast/kwen1/structured-rotations-v2/staging/20260910_191426-dff84f-edd1bf833-code --verify
  ```

- Test-only command:

  ```bash
  sbatch --test-only --no-requeue --export=NIL --account=vision-torralba-urops-meng --qos=vision-torralba-interactive --partition=vision-torralba-rtx3090 --job-name=rot-v2-gatea-a2-r1 --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=02:00:00 --gres=gpu:rtx_3090:1 --chdir=/data/scratch-fast/kwen1/structured-rotations-v2/staging/20260910_191426-dff84f-edd1bf833-code --output=/data/vision/torralba/u/kwen1/structured-rotations-v2/gate-a-execution-r1/logs/slurm-%j.out --error=/data/vision/torralba/u/kwen1/structured-rotations-v2/gate-a-execution-r1/logs/slurm-%j.out experiments/structured_rotations_v2/gate_a/run_execution.sbatch
  ```

- Sole submission and terminal-owner command: the same command with
  `--test-only` replaced by `--parsable --wait`. It returned job `1824515`.

### Failure evidence and budget

- Slurm placed job `1824515` on `torralba-3090-3`, whose recorded feature and
  GRES identify NVIDIA GeForce RTX 3090 GPUs. The application aborted before
  its own CUDA device query, so driver and runtime-device observations are not
  available from A2.
- Slurm state was `FAILED`, exit code `1:0`, elapsed `00:00:02`, with one GPU.
  Allocated A2 GPU time is `2 / 3600 = 0.000556` GPU-hours. Active GPU time is
  unavailable because the run aborted before its timer/result record.
- Retained run directory:
  `/data/vision/torralba/u/kwen1/structured-rotations-v2/gate-a-execution-r1/runs/slurm-1824515/`.
- Failure manifest SHA-256:
  `f5bfaabdb9e66c84d90b6dc75f5685f301b394be898b619d3de1ec06b0d8d627`.
- Event trace SHA-256:
  `dda19be6faf4f086439a9d52b2c0194f2a457cb3fd03cf8065dd4262e5c70824`.
- Slurm log SHA-256:
  `1e209a8851c664b571d214dd9e24a0ffe9c5c0ba029fdf7335b1d7216cf4e0c1`.
- Accounting record SHA-256:
  `deb99224278d5431548eeb406b15521915eb8ce1870e4a15e29692058d03c2c2`.
- `results.json`, raw timings, and measurement-ledger rows are absent by
  construction. The event trace contains only the verified stage identity and
  `run_started` event.

### Assessment and repair

- A1 still establishes a material full-versus-PeRQ-32 output-error gap. Job
  `1824515` establishes no A2 cost fact, so it cannot support GO,
  quality-at-fixed-cost, or STOP. Historical isolated timings and the old
  selector remain supporting context only.
- The failure is an execution-environment miss, not evidence that the native
  W4A4 backend is unavailable. The frozen A1 scratch environment has the same
  numpy, torch, transformers, tokenizers, and Triton versions plus
  `pyarrow==17.0.0`.
- Smallest isolated repair: point A2 at that read-only-compatible scratch
  environment and add a CPU dependency preflight before requesting a GPU. A
  replacement still requires a fresh full stage and explicit authorization for
  a new attempt.
- Confounds: no kernel compiled or ran; actual producer shapes were not captured;
  equal-fusion timing was not tested; the accelerator identity is from Slurm,
  not the aborted application; and PeRQ remains segment-only under the frozen
  A1 contract.

Exactly one next experiment: after authorization for a new attempt, run one
replacement of the unchanged frozen A2 workloads from a fresh complete stage
using the matching A1 scratch environment with `pyarrow==17.0.0` and a passed
CPU import preflight.

## 2026-09-10 — Replacement environment preflight and minimal repair

- No GPU was requested. Scheduler-tracked CPU preflight job `1825764` ran on
  `groenig-3` in `tig-cpu` with account `csail`, QoS `tig-main`, one CPU,
  4 GiB, and a five-minute limit. It completed `0:0` in 52 seconds.
- The exact Python environment is the unchanged A1 environment
  `/data/scratch-fast/kwen1/structured-rotations-v2/envs/gate-a-quality`.
  Python 3.10.20 imported the required frozen runtime stack: accelerate 1.14.0,
  NumPy 1.24.4, PyArrow 17.0.0, safetensors 0.8.0, tokenizers 0.22.2, Torch
  2.8.0+cu128, Transformers 5.12.1, and Triton 3.4.0.
- The CPU job loaded the preserved A1 `layer-00.pt`, rehashed it to
  `ac76a0aee1260dd9947c1cde11a4d54535d859fa277ce61c9402cfcb7e23b7d7`,
  and observed the frozen 12288-wide statistics and 1024-by-12288 BF16 uniform
  and tail activation reservoirs. CUDA was unavailable, as required for the
  CPU-only check.
- Result and Slurm-log SHA-256 values are respectively
  `d869357292dfa8d11cbb84c13275e443751d1d49698dfcc8458d3b48cc2a00b7`
  and `91ef002e427be4bb4ee38adea3e382659df11e5211fe8583bb6bb4fccd0878f6`.
  They are retained under
  `/data/vision/torralba/u/kwen1/structured-rotations-v2/gate-a-execution-r1/provenance/environment-preflight-20260910T234913Z/`.
- The preflight Python and batch-script SHA-256 values are
  `68d84b5b48599cbab0744f9984cbc6cce7d3d1d29e30051514111ac8dec533ba`
  and `dc80603069b4bc035e19403ba3050695e268b655250c6843b9451e01333190b2`.
  The batch script used `--export=NIL` and did not set or repurpose `HOME`.
- A prior CPU-only check, job `1825673`, is preserved as failed checker evidence:
  it unnecessarily required the unused `datasets` package and failed before
  cache loading. A2 reads Arrow directly through PyArrow and contains no
  `datasets` import. This was a preflight-specification correction, not an
  environment mutation or A2 attempt.
- The minimal source repair changes the A2 interpreter/PATH from the incomplete
  `causal_forcing` environment to the proven A1 environment and removes the
  unsafe `HOME` override. It changes no method, workload, tensor, quantizer,
  native backend, fusion boundary, sample count, or decision threshold.

Exactly one next experiment: create and verify a fresh complete CSAIL stage,
inspect duplicate ownership and live Slurm inventory, pass `sbatch --test-only`,
and submit the one authorized unchanged replacement A2 job.
