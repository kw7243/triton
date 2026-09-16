# Structured Rotations v2 Gate A execution decision — r2

Status: Decision A not reached. Gate B remains closed.

## Evidence

- Preserved A1 job `1818754` remains the only valid scientific result. Its
  `results.json` SHA-256 is
  `51c94593a17e855c650ff1c23518a955d9a58dd2df249ac6bd52035ea1180ee4`.
  Full Hadamard has pooled output error `0.030639`, versus `0.096933` for
  PeRQ-32 and `0.047932` for PeRQ-128.
- Historical A2 job `1824515` failed before measurement because its Python
  environment lacked PyArrow. Replacement job `1826099` failed before Python
  because its wrapper incorrectly inspected `SLURM_SUBMIT_DIR`. Both failures,
  stages, logs, and manifests remain unchanged.
- CPU-only Slurm preflight job `1825764` completed `0:0`. The unmodified A1
  environment imported the frozen stack, including `pyarrow==17.0.0`, and
  loaded the A1 layer-0 cache with SHA-256 `ac76a0ae...b7d7`.
- The captain authorized one corrected attempt. Job `1955670` was its sole
  submission, after successful `sbatch --test-only` reference `1955641`. It
  used source commit `09b970b4c526a9a1d84e236a72ad867ba8cd6ac9` and the
  immutable 1,753-entry stage
  `/data/scratch-fast/kwen1/structured-rotations-v2/staging/20260915_130019-1aa854-09b970b4c-code`.
- Terminal accounting reports `FAILED 1:0`, start/end
  `2026-09-15T23:18:53-04:00` / `2026-09-15T23:19:57-04:00`, elapsed 64
  seconds, and node `torralba-3090-3`. The allocation was one RTX 3090, four
  CPUs, 32 GiB, and two hours maximum. `squeue` and `scontrol` no longer find
  the aged terminal job; `sacct` is authoritative.
- The corrected wrapper worked. Its first event records the exact scratch stage,
  source commit/tree, and manifest hash. The application then verified one
  visible RTX 3090 with compute capability 8.6 and 25,296,044,032 bytes,
  CUDA 12.8, Torch 2.8.0+cu128, Triton 3.4.0, the frozen inputs, and producer
  shapes `[2048,4096]`, `[1,4096]`, and `[8,4096]`.
- The run failed while constructing the full-Hadamard segment runner. Triton JIT
  compilation rejected staged `execution.py:211`,
  `half: tl.constexpr = 1 << stage`, with
  `ValueError('half is already defined. constexpr cannot be reassigned.')`.
  Triton 3.4.0 statically unrolls `tl.static_range` in one local scope, so the
  annotated assignment succeeds once and is rejected on the next iteration.
- This occurred before the Hadamard kernel could dispatch, before the complete
  native-validation set, before the segment timing loop, and before all
  end-to-end work. Control flow proves the identity runner and its validation
  returned in memory first, but no validation record was persisted. The last
  event is `tensor_contract_verified`; no `native_segment_completed` event
  exists. The run produced no `results.json`, raw timings, or measurement
  ledger.
- The job log SHA-256 is
  `628e2839a2d4b21d0c576e430453cb8a8504d8072f1ac4c41e12ed0cc9d72c92`.
  The four-row event trace was preserved unchanged and mirrored into the r2 run
  directory; both copies have SHA-256
  `7a36ba8f514a541d08aaade8b9083817a20be56852012174b019864606ec180e`.
  Terminal accounting and scheduler captures are under
  `/data/vision/torralba/u/kwen1/structured-rotations-v2/gate-a-execution-r2/attempts/corrected-a2-r3-84aa6b30da/terminal/`.
- The machine-readable corrected-attempt failure manifest is
  `data/structured-rotations-v2-gate-a-execution-r2/FAILURE_MANIFEST_1955670.json`,
  SHA-256
  `c7d2a8934d21279fa07e69788219e3ca6f83d694f64d1c1ee6e1496e8a4548c6`.

## Accounting

- A1 lane allocation, including its historical wrapper failure: `0.027500`
  GPU-hours.
- A2 job `1824515`: `0.000556` allocated GPU-hours; no active timing.
- Replacement A2 job `1826099`: `0.000278` allocated GPU-hours; no active
  timing.
- Corrected A2 job `1955670`: 64 allocated GPU-seconds, or
  `0.0177777778` GPU-hours. Contract active GPU time is unavailable, not zero,
  because the process failed before it wrote that metric.
- Gate A total allocated GPU time is 166 GPU-seconds, or `0.0461111111`
  GPU-hours. Valid measured active GPU time remains A1's
  `0.0229859799061281` GPU-hours.

## Scientific decision

Do not infer GO, quality-at-fixed-cost, or scientific STOP. A1 still shows a
quality gap, but A2 produced no matched native segment or end-to-end cost
evidence. Job `1955670` is evidence of a Triton compile-time implementation
failure, not a latency or quality result for full, local, or PeRQ rotations.
A1 therefore cannot be combined with A2 into Decision A.

## Next decision boundary

The one corrected attempt has been consumed, and no retry, new Slurm action, or
Gate B experiment is authorized. The captain must choose between:

1. authorizing a new, narrowly scoped compiler repair and A2 attempt; or
2. closing Gate A as inconclusive because A2 execution evidence is unavailable.

If a repair is later authorized, its minimum scope is to remove the loop-local
constexpr rebinding without changing the scientific contract, then exercise the
affected transform under the frozen Triton 3.4.0/SM86 stack for stages 5, 7,
and 10 against the existing Torch reference before any new final submission.
