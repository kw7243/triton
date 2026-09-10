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
