# Phase A Hurwitz decode baseline

Outcome: **BLOCKED — NO RESULT**. No correctness-qualified timing row exists. Two reproducibly staged jobs were cancelled by Slurm UID 0 before the batch script produced output.

| Job | Slurm state | Exact failure | Evidence |
|---:|:---|:---|:---|
| 1516691 | false `COMPLETED 0:0` | Direct AFS helper execution denied; wrapper omitted final status propagation | `failed-job-1516691/` |
| 1516880 | `FAILED 126:0` | `/bin/bash` could not read the token-protected AFS helper | `failed-job-1516880/` |
| 1517414 | `CANCELLED by 0`, `0:0` | Slurm cancelled after 2 seconds on `torralba-3090-2`; no output file | `scheduler-cancellation-evidence.md` |
| 1524492 | `CANCELLED by 0`, `0:0` | Slurm cancelled after 6 seconds on `torralba-3090-1`; no output file | `scheduler-cancellation-evidence.md` |

Concrete root cause: `/afs` is AuristorFS. The helper's Unix mode is `775`, but the enclosing AFS ACL grants `system:anyuser` only lookup (`l`), not read. Login succeeds with kwen1 AFS tokens; the Slurm batch process has no AFS token. A tokenless `pagsh` reproduces `readable=no` and `/bin/bash` status `126`.

Minimal verified correction: invoke the exact required AFS helper on the authenticated login node. Its staged command is `sbatch`, so the helper completes the immutable full-repository copy, changes into it, exports `RESEARCH_REPRO_STAGED_DIR` and `RESEARCH_REPRO_SOURCE_REPO`, and only then submits the GPU job. The batch wrapper consumes those exported paths, requires staging metadata, refuses to run when source and stage are the same path, and runs the benchmark from the snapshot while writing to the absolute source result directory. The scratch helper mirror was only a diagnostic control and will not be used.

The exact planned command and read-only validation are recorded in `outer_submission_design.txt`.

The authorized outer invocation at commit `5d29df20b` announced a 108 MiB partial stage at `/data/scratch-fast/kwen1/compute-native-vq/staging/20260820_023734-b9ea07-5d29df20b-code`, then its tracked SSH process exited `255` before metadata or `sbatch`. No Slurm job was submitted. Evidence is in `outer_staging_failure_20260820_023734.txt`; the partial stage is preserved.

Infrastructure blocker: two independent one-GPU Torralba allocations were cancelled by Slurm UID 0 before the benchmark emitted any output. Both have `Reason=None`, zero exit and derived-exit codes, cancelled batch steps, completed extern steps, and no Slurm output file. No accessible Slurm field, node event, or comment records the cause. Correctness, tuning, timing, stability, CSV, plot, and GO/OPTIMIZE-ONCE/KILL remain unavailable.

Smallest unblock: a CSAIL Slurm administrator must inspect controller/daemon logs for jobs `1517414` and `1524492`, then clear the forced cancellation or identify an allowed Torralba account/QoS/partition invocation. Only after that external confirmation should one staged job be submitted.
