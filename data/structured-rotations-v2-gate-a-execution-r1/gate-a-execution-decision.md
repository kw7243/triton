# Gate A execution decision

Status: Decision A not reached; the authorized replacement A2 is pending after
a successful scheduler-tracked CPU environment preflight. Gate B remains closed.

## Evidence

- A1 retains a material output-error advantage for full Hadamard over PeRQ-32:
  `0.030639` versus `0.096933`. This is quality evidence only.
- The single A2 submission was Slurm job `1824515` from source commit
  `edd1bf833e6e50cb458684e077673203766ef4ea` and byte-verified stage
  `/data/scratch-fast/kwen1/structured-rotations-v2/staging/20260910_191426-dff84f-edd1bf833-code`.
- The job failed in two seconds on `torralba-3090-3` at `import pyarrow`.
  It did not reach CUDA, native-consumer validation, shape capture, segment
  timing, or end-to-end timing. No retry or requeue was submitted.
- Retained failure evidence is under
  `/data/vision/torralba/u/kwen1/structured-rotations-v2/gate-a-execution-r1/`.
  `runs/slurm-1824515/FAILURE_MANIFEST.txt` has SHA-256
  `f5bfaabdb9e66c84d90b6dc75f5685f301b394be898b619d3de1ec06b0d8d627`.
- Allocated A2 cost was `0.000556` GPU-hours. Active GPU time and all A2
  measurements are unavailable.

## Decision

Do not infer GO, quality-at-fixed-cost, or STOP. A1 supplies the required
quality contrast, but A2 supplied no matched native cost evidence. The old
isolated timings, old low-quality baseline, and rejected selector do not fill
that gap.

The failure is not evidence against the native W4A4 path. CPU-only Slurm job
`1825764` proved that the frozen A1 environment imports the complete required
A2 stack, including `pyarrow==17.0.0`, and loads the byte-verified A1 layer-0
input. The replacement changes only the environment path; all scientific A2
methods, workloads, quantizer semantics, native consumer, and timing settings
remain frozen.

Exactly one next experiment: run the authorized single replacement of the
unchanged frozen A2 workloads from a fresh complete CSAIL stage.
