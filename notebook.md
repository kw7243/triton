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

## 2026-09-15 — Replacement A2 terminal failure and r2 reconciliation

### Current state

- Sole replacement job `1826099` is terminal `FAILED 2:0`. The batch wrapper
  rejected the AFS submission directory before Python or A2 started.
- No A2 timing exists. Decision A is not reached, Gate B remains closed, and
  no second replacement is authorized.

### Preflight, source, and stage

- The authorized repair stayed limited to environment selection. Source commit
  `999b695dde0bad090104d1e619c14f6271c4b1fe`, tree
  `c28db9e3802b46eaa1af80ca7ab0213ed5172a68`, selects the unmodified A1
  environment and leaves `execution.py` byte-identical at SHA-256
  `3a0123227c33d32d31be872d7cfe783342c39957525b46743346ab92efd34890`.
- CPU-only Slurm preflight job `1825764` completed `0:0` before the GPU request.
  It imported Python 3.10.20, accelerate 1.14.0, NumPy 1.24.4, PyArrow 17.0.0,
  safetensors 0.8.0, tokenizers 0.22.2, Torch 2.8.0+cu128, Transformers
  5.12.1, and Triton 3.4.0 from
  `/data/scratch-fast/kwen1/structured-rotations-v2/envs/gate-a-quality`.
  It also loaded the frozen A1 layer-0 input on CPU and verified SHA-256
  `ac76a0aee1260dd9947c1cde11a4d54535d859fa277ce61c9402cfcb7e23b7d7`.
  The environment was not mutated.
- The fresh full stage is
  `/data/scratch-fast/kwen1/structured-rotations-v2/staging/20260910_195644-1f614e-999b695dd-code`.
  It has an ordinary `.git`, HEAD/tree `999b695dde...` / `c28db9e380...`, and a
  1,750-entry `STAGE_FILE_MANIFEST.json` with SHA-256
  `29c04a11e82a9b7cc9c0aa6a46ea20822385078e4c0600d5098f72247e1b98c0`.
  `REPRODUCIBILITY_METADATA.json` has SHA-256
  `5a2175ace0906ff8639e5455f20a43dda6fc14b92a602424467ecaeec7c02595`.
  The included native extension and A1 cache rehash to `10a961e8...e0a8e0`
  and `ac76a0ae...b7d7` respectively.
- The repair execution contract has SHA-256
  `461baeefc12291b8772d41d77248c01c905dfda6f194d69fe7ba347b7853a7f1`.
  Relative to the sealed contract, only its environment path changed. Model,
  methods, workloads, tensor shapes, quantizers, native backend, fusion
  boundaries, timing counts, and decision thresholds did not change.

Stage creation and manifest verification used:

```bash
/data/scratch-fast/kwen1/structured-rotations-v2/gate-a-execution-r1/tools/stage_and_run.sh --repo-root /data/scratch-fast/kwen1/structured-rotations-v2/gate-a-execution-r1/source/999b695dde0bad090104d1e619c14f6271c4b1fe-repo --staging-parent /data/scratch-fast/kwen1/structured-rotations-v2/staging --stage-only
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/data/scratch-fast/kwen1/structured-rotations-v2/staging/20260910_195644-1f614e-999b695dd-code /data/scratch-fast/kwen1/structured-rotations-v2/envs/gate-a-quality/bin/python -m experiments.structured_rotations_v2.gate_a.stage_manifest --root /data/scratch-fast/kwen1/structured-rotations-v2/staging/20260910_195644-1f614e-999b695dd-code --verify
```

### Sole replacement submission and terminal state

- The atomic latch records zero prior replacement attempts, successful
  `sbatch --test-only` ID `1826065`, and exactly one reserved replacement.
- Exact submission argv was:

  ```bash
  /usr/bin/sbatch --parsable --wait --no-requeue --export=NIL --account=vision-torralba-urops-meng --qos=vision-torralba-interactive --partition=vision-torralba-rtx3090 --job-name=rot-v2-gatea-a2-r2 --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=02:00:00 --gres=gpu:rtx_3090:1 --chdir=/data/scratch-fast/kwen1/structured-rotations-v2/staging/20260910_195644-1f614e-999b695dd-code --output=/data/vision/torralba/u/kwen1/structured-rotations-v2/gate-a-execution-r1/logs/slurm-%j.out --error=/data/vision/torralba/u/kwen1/structured-rotations-v2/gate-a-execution-r1/logs/slurm-%j.out /data/scratch-fast/kwen1/structured-rotations-v2/staging/20260910_195644-1f614e-999b695dd-code/experiments/structured_rotations_v2/gate_a/run_execution.sbatch
  ```

- That sole replacement was job `1826099`. The terminal recheck used `squeue`,
  `sacct`, and `scontrol` for that exact ID. `squeue` and `scontrol` no longer
  found an active job; `sacct` reported `FAILED 2:0`, start/end
  `2026-09-14T20:09:11-04:00` / `2026-09-14T20:09:12-04:00`, elapsed one
  second, and node `torralba-3090-1`.
- The retained log contains only the Slurm CPU-binding line and
  `refusing non-stage submit directory: /afs/csail.mit.edu/u/k/kwen1`.
  The submission was invoked from AFS. `--chdir` selected the stage as Slurm's
  work directory, but did not rewrite `SLURM_SUBMIT_DIR`, so the batch wrapper
  exited at its stage guard before Python or the A2 application began.
- There is no run directory in the original r1 retained root and no
  `results.json`, raw timing file, event trace, native validation, tensor-shape
  capture, or measurement ledger. The failure is a submission-wrapper error,
  not a scientific result and not evidence against the repaired environment or
  native W4A4 backend.

### Retained evidence, accounting, and decision

- Canonical r2 evidence is under
  `/data/vision/torralba/u/kwen1/structured-rotations-v2/gate-a-execution-r2/`.
  The terminal failure manifest is mirrored from
  `data/structured-rotations-v2-gate-a-execution-r2/FAILURE_MANIFEST.json` and
  has SHA-256
  `13b4431e717f4ef54d8157d589bc6344fc99c930eb8578a74133967e3fe91236`.
- The canonical decision and r2 ledger have SHA-256 values
  `c503f23e3e5b4488af1d02675a17ce78bed4ba024382ea7c70e64eaa2e4f5f7d`
  and `49732c4b843b5685c5ee572baed1af656ea55def4afda20ffeb7df76f2288ae7`.
- Retained SHA-256 values are: job log `73ecbcb8...0c4a4`, terminal accounting
  `a6e500e2...9210`, scheduler capture `425a65d2...2663`, submission manifest
  `93aabcb1...a8419`, and latch contract `72cbee0e...93e4`. The successful CPU
  preflight result/log remain byte-identical at `d8693572...00b7` and
  `91ef002e...8f6`.
- Job `1826099` allocated one GPU for one second, or `0.000278` GPU-hours.
  Its active GPU time is unavailable because the application never started.
  Including the A1 lane (`0.027500`) and failed A2 jobs `1824515`
  (`0.000556`) and `1826099` (`0.000278`), Gate A allocated `0.028333`
  GPU-hours. Valid measured active time remains A1's `0.02298598` GPU-hours.
- Preserved A1 job `1818754` remains valid: full Hadamard pooled output error
  is `0.030639`, PeRQ-32 is `0.096933`, and PeRQ-128 is `0.047932`. Neither A2
  attempt produced native segment or end-to-end cost evidence, so Decision A
  is not reached and Gate B remains closed. This is not a scientific STOP.

No next experiment is authorized. The sole replacement has been consumed; a
second replacement, changed workload, bridge, selector, or Gate B run is outside
this task.

## 2026-09-15 — Captain-authorized corrected A2 attempt

### Authority and fixed scope

- The captain authorized exactly one corrected A2 attempt in
  `/home/ubuntu/firstmate/data/structured-rotations-v2-gate-a-execution-r2/corrected-a2-retry-decision.md`,
  SHA-256 `3bf7d2f004eb9b39521dad25c605f4fda43fd370376c9af4062db0763ec8329a`.
- Attempt identity is `corrected-a2-r3-84aa6b30da`. Preserved A1 job `1818754`
  and failed A2 jobs `1824515` and `1826099`, their stages, logs, manifests,
  latches, and retained outputs remain historical evidence and were not changed.
- The only production-code change is commit
  `84aa6b30da19712a1bcfd605a8a1b0fd34cc7d6f`: the batch wrapper derives the
  immutable stage with `pwd -P` instead of `SLURM_SUBMIT_DIR`. The existing
  `*-code`, staging-parent, ordinary `.git`, reproducibility-metadata, and
  stage-manifest checks remain in place.
- The faulty static assertion was reversed. An executable regression runs the
  actual wrapper guard with the process working directory set to a synthetic
  stage and `SLURM_SUBMIT_DIR` set to a different submission origin. It also
  proves that reversing those paths is rejected.
- Wrapper and test SHA-256 values are
  `d438cc5ef009db78f45c04a86466628b54d4683dbab6197dbc057ccfdaba23c9`
  and `ba67a2dce32871567813b551b81651d92c7b46658a2fd4c1e6269dd16548c5d3`.
  `execution.py` remains byte-identical at
  `3a0123227c33d32d31be872d7cfe783342c39957525b46743346ab92efd34890`;
  `execution_contract.json` remains byte-identical at
  `461baeefc12291b8772d41d77248c01c905dfda6f194d69fe7ba347b7853a7f1`.
- `/bin/bash -n`, all seven Gate A execution unit tests, and
  `git diff --check` passed. `make` was not run because no native/compiler code
  changed.

### Non-GPU preflight and resource choice

- The temporary Slurm policy through 2026-09-26 requires all non-GPU checks
  first and permits the final reproducible attempt as one `sbatch`. No debug
  allocation or extra preflight job is planned.
- Scheduler-tracked CPU preflight job `1825764` remains the clean preflight.
  Its unchanged result and log rehash to `d8693572...00b7` and
  `91ef002e...8f6`. It imported Python 3.10.20, PyArrow 17.0.0, NumPy 1.24.4,
  Torch 2.8.0+cu128, Transformers 5.12.1, tokenizers 0.22.2, and Triton 3.4.0,
  then loaded the frozen A1 layer-0 input at `ac76a0ae...b7d7`.
- The 2026-09-15T16:52Z live inventory covered both user associations and every
  visible allowed GPU partition. Torralba exposes V100, RTX 3090, H100, and
  H200. Vision Shared exposes RTX 2080 Ti, Titan RTX, A6000, A100, L40S, H100,
  H200, V100, RTX 3090, RTX 3080, RTX 6000 Ada, and RTX 4090; the `csail`
  association also exposes shared H200 and L40S.
- The frozen application itself requires compute capability 8.6 and a device
  name containing `RTX 3090`. V100/Titan/2080 Ti are the wrong architecture;
  H100/H200/L40S/Ada/4090 are not SM86; A100 and A6000 would be rejected by the
  frozen runtime identity check; RTX 3080 is both name-incompatible and too
  small. The only compatible class is one 24 GiB RTX 3090.
- At capture time all Torralba RTX 3090 GPUs and both usable shared
  `improbablex001` RTX 3090 GPUs were allocated; the other shared RTX 3090 nodes
  were drained or invalid. The dedicated Torralba tuple retains the same
  smallest device with the proven priority-100 interactive QoS; shared access
  would use priority-1 `shared-if-available` on the same or currently unavailable
  hardware.
- The selected frozen minimum is one node, one task, four CPUs, 32 GiB RAM, one
  `gpu:rtx_3090`, and two hours in `vision-torralba-rtx3090`, account
  `vision-torralba-urops-meng`, QoS `vision-torralba-interactive`. This exactly
  matches the unchanged A2 contract.

### Pre-stage reconciliation

- Fresh provenance root:
  `/data/vision/torralba/u/kwen1/structured-rotations-v2/gate-a-execution-r2/attempts/corrected-a2-r3-84aa6b30da/`.
  The timestamped inventory, associations, QoS, partition/node controls, storage
  mounts, queue, accounting, retained inventory, prior latch inventory, and
  per-file capture hashes are under `prestage/`.
- The matching live queue was empty. Matching accounting contained only terminal
  jobs `1824515` (`FAILED 1:0`) and `1826099` (`FAILED 2:0`). The only current
  user job was unrelated job `1955305`; no accepted A2 result or corrected-attempt
  latch existed.
- Both storage roots are user-owned and mounted at their declared NFS locations.
  The fresh source and immutable stage will remain under scratch; attempt
  provenance and accepted output will remain under the r2 retained root.

Exactly one next experiment: transfer the clean committed source, create and
verify one fresh full-repository stage, rerun the duplicate check, pass
`sbatch --test-only`, then issue the sole corrected A2 `sbatch` and retain its
terminal result. No Gate B work is authorized.
