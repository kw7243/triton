# Phase A Hurwitz decode baseline

Outcome: **NO RESULT YET**. Jobs `1516691` and `1516880` both failed before staging or benchmark execution; no correctness-qualified timing row exists.

| Job | Slurm state | Exact failure | Evidence |
|---:|:---|:---|:---|
| 1516691 | false `COMPLETED 0:0` | Direct AFS helper execution denied; wrapper omitted final status propagation | `failed-job-1516691/` |
| 1516880 | `FAILED 126:0` | `/bin/bash` could not read the token-protected AFS helper | `failed-job-1516880/` |

Concrete root cause: `/afs` is AuristorFS. The helper's Unix mode is `775`, but the enclosing AFS ACL grants `system:anyuser` only lookup (`l`), not read. Login succeeds with kwen1 AFS tokens; the Slurm batch process has no AFS token. A tokenless `pagsh` reproduces `readable=no` and `/bin/bash` status `126`.

Minimal verified correction: invoke the exact required AFS helper on the authenticated login node. Its staged command is `sbatch`, so the helper completes the immutable full-repository copy, changes into it, exports `RESEARCH_REPRO_STAGED_DIR` and `RESEARCH_REPRO_SOURCE_REPO`, and only then submits the GPU job. The batch wrapper consumes those exported paths, requires staging metadata, refuses to run when source and stage are the same path, and runs the benchmark from the snapshot while writing to the absolute source result directory. The scratch helper mirror was only a diagnostic control and will not be used.

The exact planned command and read-only validation are recorded in `outer_submission_design.txt`.

No further job has been submitted. Correctness, tuning, timing, stability, CSV, plot, and GO/OPTIMIZE-ONCE/KILL remain unavailable.
