# Phase A Hurwitz decode baseline

Outcome: **NO RESULT**. Slurm job `1516691` allocated the requested RTX 3090 and captured hardware/software, but the AFS staging helper failed with `Permission denied` before staging or benchmark execution.

| Requirement | State | Evidence |
|:--|:--|:--|
| Torralba one-GPU Slurm allocation | PASS | `system.txt`, `slurm_accounting.txt` |
| Hardware/software captured before benchmark | PASS | `system.txt` |
| Full repository staged inside allocation | FAIL | `stage_and_run.log`; blank `staged_snapshot.txt`; metadata absent |
| Exhaustive and random correctness | NOT RUN | `correctness.json` absent |
| J gather, F/H axis, error, finite, bf16 gates | NOT RUN | benchmark never started |
| Fixed tuning set compiled/selected before timing | NOT RUN | `tuning.json` absent |
| Cold-L2 and steady p20/p50/p80; five trials | NOT RUN | `timings.csv` and `trial_timings.json` absent |
| J/H stability and adaptive repetition | NOT RUN | no timing rows |
| Gchunks/s, output GiB/s, configurations | NOT RUN | `timings.csv` absent |
| Speedup plot | NOT RUN | `jh_speedup.png` absent |
| GO / OPTIMIZE ONCE / KILL decision | UNAVAILABLE | no correctness-qualified primary row |

The job ran on shared `torralba-3090-2` with an NVIDIA GeForce RTX 3090 (compute capability 8.6), driver 580.178.04, CUDA 13.0, PyTorch 2.13.0+cu130, and Triton 3.7.1. Slurm recorded `COMPLETED 0:0` because the wrapper did not propagate the helper failure; the wrapper is corrected after this run but was not resubmitted.

No CSV values or plot were fabricated from the failed run.
