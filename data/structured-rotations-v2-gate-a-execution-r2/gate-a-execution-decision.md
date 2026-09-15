# Structured Rotations v2 Gate A execution decision — r2

Status: Decision A not reached. Gate B remains closed.

## Evidence

- Preserved A1 job `1818754` remains the only valid quality result. Its
  `results.json` SHA-256 is
  `51c94593a17e855c650ff1c23518a955d9a58dd2df249ac6bd52035ea1180ee4`.
  Full Hadamard has pooled output error `0.030639`, versus `0.096933` for
  PeRQ-32 and `0.047932` for PeRQ-128.
- Historical A2 job `1824515` failed before measurement because its Python
  environment lacked PyArrow. Its stage, log, provenance, and failure manifest
  remain unchanged.
- CPU-only Slurm preflight job `1825764` completed `0:0`. The unmodified A1
  environment imported the frozen stack, including `pyarrow==17.0.0`, and
  loaded the A1 layer-0 cache with SHA-256 `ac76a0ae...b7d7`.
- The sole authorized replacement was job `1826099`, submitted with
  `--parsable --wait --no-requeue --export=NIL` after test-only job `1826065`.
  It used source commit `999b695dde0bad090104d1e619c14f6271c4b1fe` and the
  1,750-entry immutable stage
  `/data/scratch-fast/kwen1/structured-rotations-v2/staging/20260910_195644-1f614e-999b695dd-code`.
- Job `1826099` failed `2:0` after one allocated GPU-second on
  `torralba-3090-1`. The batch wrapper observed
  `SLURM_SUBMIT_DIR=/afs/csail.mit.edu/u/k/kwen1` and rejected it as a
  non-stage directory. Slurm's `--chdir` correctly set the work directory to
  the immutable stage, but it does not rewrite `SLURM_SUBMIT_DIR`.
- The wrapper exited before Python, CUDA inspection, model loading, native
  validation, tensor-shape capture, or timing. No run directory, results,
  raw timings, or measurement ledger was produced.
- The replacement failure manifest has SHA-256
  `13b4431e717f4ef54d8157d589bc6344fc99c930eb8578a74133967e3fe91236`.
  Canonical retained evidence is under
  `/data/vision/torralba/u/kwen1/structured-rotations-v2/gate-a-execution-r2/`.

## Accounting

- A1 lane allocation, including its historical wrapper failure: `0.027500`
  GPU-hours.
- A2 job `1824515`: `0.000556` allocated GPU-hours; no active timing.
- Replacement A2 job `1826099`: `0.000278` allocated GPU-hours; no active
  timing.
- Gate A total allocated GPU time: `0.028333` hours. Valid measured active GPU
  time remains A1's `0.02298598` hours.

## Decision

Do not infer GO, quality-at-fixed-cost, or scientific STOP. A1 still shows a
quality gap, but neither A2 attempt produced matched native segment or
end-to-end cost evidence. The replacement failure is a batch-submission
integration error, not evidence about the frozen methods, PyArrow repair, or
native W4A4 backend.

The one authorized replacement has been consumed. No second replacement or
Gate B experiment is authorized. Any further recovery requires a new explicit
decision outside this task.
