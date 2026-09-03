# Compute Native VQ research notebook

## 2026-08-20 — Phase A setup

Current state:

- Scope: independent paper-spec synthetic reconstruction of the HQMQ Phase A J/F/H decoder comparison; no unreleased HQMQ source is used.
- Repository: `/data/scratch-fast/kwen1/compute-native-vq/triton`.
- Origin: `https://github.com/kw7243/triton.git`.
- Upstream: `https://github.com/triton-lang/triton.git`; push is disabled.
- License: Triton MIT license at `LICENSE`; copied upstream source, if any, retains that notice.
- Feature branch: `fm/phase-a-hurwitz-decode`.
- Reference branch: `fm/phase-a-triton-reference`.
- Base commit: Triton v3.7.1 `f797708c0626e5f9840ca5b0a98790e2c7cb09ad`.
- Worktree was clean at the pinned commit before this notebook was added.
- Isolated environment: `/data/scratch-fast/kwen1/compute-native-vq/env/phase-a`.
- Pip cache: `/data/scratch-fast/kwen1/compute-native-vq/cache/pip`.
- Triton cache: `/data/scratch-fast/kwen1/compute-native-vq/cache/triton`.
- Result layout: `results/2026-08-20-hurwitz-decode-baseline/`.
- No experiment command or GPU code has run yet.

Live Slurm facts:

- Login host: `slurm-login-0.csail.mit.edu`.
- Account: `vision-torralba-urops-meng`.
- QoS: `vision-torralba-interactive`.
- Partition: `vision-torralba-rtx3090` (UP; nodes `torralba-3090-[1-3]`; `gpu:rtx_3090:7` per node).
- Allocation request: one GPU, four CPUs, 16 GiB host memory, one hour.
- GPU experiments will run only through Slurm from a timestamped full-repository snapshot.

Decoder assumptions:

- Quaternion order is scalar-first `(w,x,y,z)`.
- Joint identifiers use `id = p*S + s`, with the 24 primary Hurwitz units ordered by the benchmark.
- Shapes are roles `2`, KV heads `8`, head dimension `128`, deterministic seed `0`, and `int32` joint IDs shared by J/F/H.

Next step:

- Run CPU-only syntax/interface validation, commit the implementation, then stage and submit exactly once.

Implementation scope:

- `experiments/phase_a_decode/benchmark.py` is 437 lines after removing the 1,049-line first draft before commit.
- The scout's 150–250-line estimate covered the three kernel bodies and a basic timing wrapper. The acceptance contract also requires exhaustive and random correctness for two dtypes and both launch configs, an independent gather and float32 oracle, two benchmark APIs with three quantiles, five randomized trials, adaptive repetition, configuration selection, six result rows, CSV/Markdown/JSON output, and a plot. Those executable gates account for the unavoidable excess.
- The implementation remains one module: one constexpr-specialized kernel, compact correctness/timing functions, and artifact writers. There is no framework, plugin, reusable benchmark abstraction, H-cache, or speculative optimization.
- The only additional executable source is the reproducible Slurm entry point.

Environment setup:

- System `python3 -m venv` failed before package installation because `ensurepip` is unavailable.
- Preserved partial prefix: `/data/scratch-fast/kwen1/compute-native-vq/env/phase-a.ensurepip-failed-2026-08-20`.
- Created the required prefix with scratch micromamba: Python `3.11.15` at `/data/scratch-fast/kwen1/compute-native-vq/env/phase-a`.
- Installed PyTorch `2.13.0`, Triton `3.7.1`, and Matplotlib `3.10.5`; the job will record the complete resolved package set.
- Initialized the repository's no-mistakes gate with fork URL `https://github.com/kw7243/triton.git`; the shared daemon is running and was not modified.
- No experiment command or GPU code has run yet.

Staging attempts:

- Attempt 1 invoked the required helper from the source repo for commit `9047bd25c7f88ef71c8a8fbdc71c4e1bf36f4a25`.
- Incomplete directory: `/data/scratch-fast/kwen1/compute-native-vq/staging/20260820_014155-632901-9047bd25c-code` (107 MiB).
- The SSH session exited `255` during `rsync`; the helper did not print `Staging complete`, and `REPRODUCIBILITY_METADATA.json` is absent.
- `squeue -u kwen1` was empty immediately afterward, proving that no experiment job was submitted.
- Next action: retry the helper once with transport keepalives; preserve this failed directory as evidence.

- Attempt 2 invoked the same helper with SSH keepalives for commit `46e810f3c04895ef61fc2344f41c1d235bb2e30d`.
- Incomplete directory: `/data/scratch-fast/kwen1/compute-native-vq/staging/20260820_014338-57c0e6-46e810f3c-code` (107 MiB).
- The connection again exited `255` during `rsync`; metadata is absent and no job was submitted.
- Both partial directories are preserved and will not be deleted.

Resolved execution path:

- Firstmate identified login-session termination during the 107 MiB copy as the cause.
- Submit `experiments/phase_a_decode/run_phase_a.sbatch` once directly from the source repo.
- Inside the one-GPU allocation, capture hardware/software first, then invoke the exact AFS `stage_and_run.sh` with staging parent `/data/scratch-fast/kwen1/compute-native-vq/staging` and the Phase A Python command.
- The helper changes into its completed snapshot before running the benchmark; the benchmark writes to the absolute source result directory.

## 2026-08-20 — Slurm job 1516691

Current state:

- Submitted wrapper commit: `4bddbcb4f17a93b77f36cdb2b08281228cc4d907`.
- Allocation: job `1516691`, account `vision-torralba-urops-meng`, QoS `vision-torralba-interactive`, partition `vision-torralba-rtx3090`, one GPU, four CPUs, 16 GiB host-memory request.
- Slurm accounting: `COMPLETED`, exit `0:0`, elapsed `00:00:07`, node `torralba-3090-2`, start `2026-08-20T02:08:18`, end `2026-08-20T02:08:25`.
- Hardware: NVIDIA GeForce RTX 3090, compute capability `8.6`, 24,576 MiB VRAM, driver `580.178.04`, CUDA runtime `13.0`.
- Software: Python `3.11.15`, PyTorch `2.13.0+cu130`, Triton `3.7.1`; bf16 reported supported.
- Host memory: 251 GiB total, 216 GiB available at capture.
- Node was shared: six other jobs were recorded alongside job `1516691`.

Failure:

- Hardware/software capture completed before benchmark code.
- Direct execution of `/afs/csail.mit.edu/u/k/kwen1/.codex/skills/research-reproducibility/scripts/stage_and_run.sh` failed with `Permission denied` on the allocated node.
- The helper is mode `775` on the login node; the allocated-node failure requires invoking the readable script through `bash`.
- The wrapper omitted its intended final `exit "$status"`, so Slurm recorded false-success `0:0` after the helper failure.
- No completed staged snapshot or `REPRODUCIBILITY_METADATA.json` exists. `staged_snapshot.txt` contains only a newline.
- The benchmark never started. `timings.csv`, `correctness.json`, `tuning.json`, `trial_timings.json`, `run_metadata.json`, and `jh_speedup.png` are absent.
- Correctness, bf16 execution, tuning selection, cold/steady timings, stability, throughput, and the primary decision gate are therefore unvalidated.
- Gate decision: **NO RESULT**. This is not GO, OPTIMIZE ONCE, or KILL evidence.

Artifacts:

- Result directory: `/data/scratch-fast/kwen1/compute-native-vq/triton/results/2026-08-20-hurwitz-decode-baseline/`.
- Preserved job evidence directory: `results/2026-08-20-hurwitz-decode-baseline/failed-job-1516691/`.
- Evidence there includes `system.txt`, `environment.lock.txt`, `command.txt`, `stage_and_run.log`, `slurm-1516691.out`, blank `staged_snapshot.txt`, `slurm_accounting.txt`, `artifact_hashes.sha256`, and the failed-run `README.md`.
- Preserved incomplete login-stage directories remain the two paths recorded above.
- Firstmate authorized one replacement after the failed job ran no experiment.
- Minimal wrapper correction: invoke the exact helper as `/bin/bash "$STAGING_HELPER" --staging-parent ...`, retain the required benchmark command, and end with `exit "$status"` so helper or benchmark failures propagate to Slurm.
- No benchmark logic or experiment scope changed.
- Next action: commit this evidence/fix, then submit exactly one corrected wrapper.

## 2026-08-20 — Slurm job 1516880 diagnosis

Current state:

- Submitted commit: `b3eb35bd959fc6568170fc2e13ea1e725097673c`.
- Slurm: job `1516880`, `FAILED`, exit `126:0`, elapsed `00:00:07`, node `torralba-3090-1`.
- Exact failing command: `/bin/bash "$STAGING_HELPER" --staging-parent "$STAGING_PARENT" -- <benchmark command>`.
- Saved error: `/bin/bash: /afs/csail.mit.edu/u/k/kwen1/.codex/skills/research-reproducibility/scripts/stage_and_run.sh: Permission denied`.
- No staged snapshot, metadata, or benchmark artifact was created; no experiment ran.
- Evidence directory: `results/2026-08-20-hurwitz-decode-baseline/failed-job-1516880/`.

Root cause:

- `/afs` is mounted as `auristorfs` from source `AFS`.
- The helper and path components show Unix mode `775`, but the effective AFS ACL grants `kwen1 rlidwka` and `system:anyuser l` only.
- The login process holds kwen1 rxgk/rxkad AFS tokens, so it can read and execute the helper.
- A fresh tokenless `pagsh` has no tokens, reports the AFS helper `readable=no`, and reproduces `/bin/bash ... --help` status `126`.
- Slurm batch jobs did not inherit the login AFS token; invoking `/bin/bash` cannot bypass the missing AFS read right.

Minimal verified correction:

- The scratch helper mirror was useful only as a control proving the helper bytes themselves are valid. It does not satisfy the exact-AFS-path requirement and will not be used for an experiment.
- Invoke `/afs/csail.mit.edu/u/k/kwen1/.codex/skills/research-reproducibility/scripts/stage_and_run.sh` directly on the authenticated login node, where `/bin/bash ... --help` returns `0`.
- Pass `sbatch --export=ALL,SOURCE_REPO=/data/scratch-fast/kwen1/compute-native-vq/triton experiments/phase_a_decode/run_phase_a.sbatch` as the helper's staged command.
- The helper source shows that it completes metadata, changes to the staged repository, exports `RESEARCH_REPRO_STAGED_DIR` and `RESEARCH_REPRO_SOURCE_REPO`, then executes `sbatch` there.
- The wrapper no longer calls any helper. It consumes those exported paths, refuses the mutable source path, requires `REPRODUCIBILITY_METADATA.json`, captures hardware/software in Slurm, and runs the benchmark from the snapshot. `set -euo pipefail` propagates setup or benchmark failure as a nonzero Slurm exit.
- Read-only validation: helper help status `0`; wrapper `bash -n` status `0`; no `stage_and_run` or `STAGING_HELPER` reference in the wrapper; source-equals-stage rejection status `125`; `sbatch --test-only` status `0` under the verified account/QoS/partition.
- The exact planned command and proof are in `results/2026-08-20-hurwitz-decode-baseline/outer_submission_design.txt`.
- No further Slurm job has been submitted. Report this correction before any submission.

## 2026-08-20 — Authorized outer staging attempt

- Authorized source commit: `5d29df20bc188ffb2196de4994c322966b85403b`; source branch was clean.
- Invoked the exact AFS helper on the authenticated login node with the exact staged `sbatch` command recorded in `outer_submission_design.txt`.
- The helper announced target `/data/scratch-fast/kwen1/compute-native-vq/staging/20260820_023734-b9ea07-5d29df20b-code`.
- The tracked foreground SSH process exited `255` during repository copy, before `Staging complete` or any Slurm job ID.
- Inspection at `2026-08-20T02:38:43-04:00`: target exists and is 108 MiB; `REPRODUCIBILITY_METADATA.json` is absent; no matching helper or rsync process remains.
- `squeue -u kwen1` was empty and Slurm accounting contained no submission after 02:30, so no job or experiment was launched.
- The partial stage is preserved and was not deleted.
- Evidence: `results/2026-08-20-hurwitz-decode-baseline/outer_staging_failure_20260820_023734.txt`.
- This is the third authenticated login staging copy terminated before metadata. Per the retry gate, do not resubmit without Firstmate intervention.

## 2026-08-21 — Compact staging recovery and scheduler blocker

Staging recovery:

- Authenticated tmux attempts r1–r5 and every partial stage remain preserved under `/data/scratch-fast/kwen1/compute-native-vq/run-state/` and `/data/scratch-fast/kwen1/compute-native-vq/staging/`; none were deleted.
- Full-worktree staging repeatedly hit a roughly 54-second login execution limit before metadata, including r5 after shrinking history to a 7.04 MiB pack.
- Verified r5 run source: `/data/scratch-fast/kwen1/compute-native-vq/run-source/phase-a-hurwitz-decode-b8d099336-r5`, clean shallow branch `fm/phase-a-hurwitz-decode`, HEAD `b8d099336ca65cf7aaa1a1b664980f19f5e4e474`, tree `89d495c8884a9ec51187238d6da06123908591e1`, 1,492 tracked files, no alternates, 28 MiB total.
- Verified r6 bare capsule: `/data/scratch-fast/kwen1/compute-native-vq/run-source/phase-a-hurwitz-decode-b8d099336-r6.git`, same HEAD/tree, bare and shallow, fsck-clean, no alternates, 24 files, one independent 7.04 MiB pack.
- The exact AFS helper copied each compact bare capsule before its staged command materialized a clean 1,492-file worktree, copied `REPRODUCIBILITY_METADATA.json`, and submitted from that worktree with explicit staged/source paths.

Scheduler terminal records:

- Job `1517414`: `CANCELLED by 0`, exit `0:0`, derived exit `0:0`, reason `None`; start `2026-08-20T08:37:37`, end `08:37:39`, node `torralba-3090-2`. Batch step cancelled; extern step completed. No `slurm-1517414.out` exists.
- Job `1524492`: guarded replacement, also `CANCELLED by 0`, exit `0:0`, derived exit `0:0`, reason `None`; start `2026-08-21T00:16:44`, end `00:16:50`, node `torralba-3090-1`. Batch step cancelled; extern step completed. No `slurm-1524492.out` exists.
- No Phase A job remains active. No benchmark output or correctness/timing artifact exists.
- Node context did not identify a cause: contemporaneous jobs completed on `torralba-3090-2`, Slurm records no reason/comment or node event, and controller/daemon logs are not readable by this account.

Preserved staged worktrees:

- Job `1517414`: `/data/scratch-fast/kwen1/compute-native-vq/staging/20260820_031814-56491e-b8d0993-code/worktree`.
- Job `1524492`: `/data/scratch-fast/kwen1/compute-native-vq/staging/20260820_193950-fad1f6-b8d0993-code/worktree`.
- Both are clean at HEAD `b8d099336ca65cf7aaa1a1b664980f19f5e4e474`, tree `89d495c8884a9ec51187238d6da06123908591e1`, with 1,492 tracked files and copied reproducibility metadata.
- Full evidence and hashes: `results/2026-08-20-hurwitz-decode-baseline/scheduler-cancellation-evidence.md`.

Gate:

- **BLOCKED — NO RESULT.** The repeated obstacle is a scheduler/controller-side UID 0 cancellation before the batch script opens its output, not a correctness or performance result.
- Smallest unblock: a CSAIL Slurm administrator inspects slurmctld/slurmd logs for `1517414` and `1524492` and clears the forced cancellation or identifies an allowed Torralba account/QoS/partition invocation. Do not submit again until that external action occurs.

## 2026-08-24 — Captain-authorized controlled retry 3

Authorization and ownership:

- Captain decision `corr=f633238db1c8d4e2` authorizes exactly one controlled third Torralba submission despite unchanged UID-0 cancellation evidence. It supersedes the earlier no-third-attempt rule only for this attempt.
- `phase-a-hurwitz-decode-p1` exclusively owns scheduler context, fresh whole-repository staging, submission, provenance, and the sole terminal-state event source.
- `phase-a-retry3-validation` independently owns source/configuration audit and result validation; it must not submit, cancel, or poll. Scheduler ownership will not duplicate its analysis.

Exact source reconciliation before this notebook edit:

- Repository: `/data/scratch-fast/kwen1/compute-native-vq/triton`; branch `fm/phase-a-hurwitz-decode`.
- HEAD `7b3457b1fac80850e0ba4addfc8f124b023388f6`; tree `73c94ff2e50187f720a378fa790b4101658c3347`.
- Status: clean, `fm/phase-a-hurwitz-decode...origin/fm/phase-a-hurwitz-decode [ahead 8]`; no reset, discard, or prior-stage modification.
- Remotes: `origin=https://github.com/kw7243/triton.git`; `upstream=https://github.com/triton-lang/triton.git`.
- Batch SHA-256 `aca356cc4d1ba87ea67c94f6bca671e52e0a26b74a779722fae218dd01ba8bc6`; benchmark SHA-256 `c2c451ab3290af8d4fc97ff8bec077533a679ffd825a2d44d8907dc8cdd16a6a`.

Scheduler and storage preflight:

- Checked from `slurm-login-0.csail.mit.edu` as `kwen1`; `SLURM_JOB_ID` was unset and no `/dev/nvidia*` devices were visible. No GPU or benchmark code ran on the login node.
- Current association permits account `vision-torralba-urops-meng` and QoS `vision-torralba-interactive`; partition `vision-torralba-rtx3090` is `UP`, exposes only `torralba-3090-[1-3]`, and currently has two idle nodes and one mixed node.
- `run_phase_a.sbatch` requests one node and exactly one GPU with `--gres=gpu:1`, 4 CPUs, 16 GiB, and one hour. Its absolute stdout/stderr path is `$RESULT_DIR/slurm-%j.out`, where `$RESULT_DIR` is `/data/scratch-fast/kwen1/compute-native-vq/triton/results/2026-08-20-hurwitz-decode-baseline`.
- `bash -n` and `sbatch --test-only` passed. The dry run mentioned projected ID `1569898`; immediate `squeue` and `sacct` checks proved that ID does not exist, so it was not a submission.
- Prior preserved scheduler records remain jobs `1517414` and `1524492`, both `CANCELLED by 0`; their staged worktrees and all earlier partial stages/evidence remain untouched.

Reproducible launch contract:

- Invoke the exact AFS helper from this source repository: `/bin/bash /afs/csail.mit.edu/u/k/kwen1/.codex/skills/research-reproducibility/scripts/stage_and_run.sh --repo-root /data/scratch-fast/kwen1/compute-native-vq/triton --staging-parent /data/scratch-fast/kwen1/compute-native-vq/staging -- sbatch --parsable --account=vision-torralba-urops-meng --qos=vision-torralba-interactive --partition=vision-torralba-rtx3090 --export=ALL,SOURCE_REPO=/data/scratch-fast/kwen1/compute-native-vq/triton experiments/phase_a_decode/run_phase_a.sbatch`.
- The helper must create a fresh full-repository snapshot and `REPRODUCIBILITY_METADATA.json`, then execute `sbatch` from inside that snapshot with `RESEARCH_REPRO_STAGED_DIR` exported by the helper.
- Expected durable outputs: `slurm-<jobid>.out`, `staged_snapshot.txt`, `REPRODUCIBILITY_METADATA.json`, `environment.lock.txt`, `system.txt`, `command.txt`, `run_metadata.json`, `correctness.json`, `tuning.json`, `trial_timings.json`, `timings.csv`, updated `README.md`, and `jh_speedup.png` under the absolute result directory above.
- After submission, record the fresh stage, metadata, exact job ID/command/paths here and commit on this feature branch. Register exactly one authenticated terminal-state check under task identity `phase-a-hurwitz-decode-p1`.
- **No fourth attempt is authorized.** If this job is cancelled by UID 0 or otherwise fails, do not submit automatically.

Staging outcome:

- At `2026-08-24T19:09:34-04:00`, the scheduler owner invoked the exact AFS helper and staged `sbatch` command above once from the clean source commit `18822c75be1e4359013e6b9771c0b2d089cba70c`, tree `560b950920bed61fc3ca05997568d2aca80afde0`.
- The helper announced fresh target `/data/scratch-fast/kwen1/compute-native-vq/staging/20260824_190934-b13411-18822c75b-code`.
- The tracked foreground SSH command ended with status `255` after about 52 seconds, before `Staging complete`, metadata creation, or a numeric `sbatch` ID.
- The incomplete 108 MiB stage is preserved unchanged. It contains 1,615 files and resolves to the expected HEAD/tree, but `REPRODUCIBILITY_METADATA.json` is absent.
- No matching helper/rsync process remained at `2026-08-24T19:11:05-04:00`.
- `squeue -u kwen1` was empty and `sacct` contained no `phase-a-hurwitz` submission after `19:08`; therefore `sbatch` did not execute and no retry-3 Slurm job exists.
- No Slurm output or benchmark/result artifact was created by this attempt.
- No terminal-state check was created or registered because there is no job ID to observe.
- **BLOCKED BEFORE SUBMISSION.** Do not bypass or repeat staging, do not submit directly, and do not create an automatic fourth attempt. Captain/supervisor direction is required for any further action.

## 2026-08-24 — Retry 3 compact recovery and single submission

Supervisor recovery and source:

- The supervisor confirmed that the failed full-worktree transport submitted no Slurm job, so the captain-authorized experiment attempt remained unused, and authorized one reuse of the proven r6/r7 compact-capsule transport. The incomplete `/data/scratch-fast/kwen1/compute-native-vq/staging/20260824_190934-b13411-18822c75b-code` stage remains untouched.
- Exact submitted source: clean branch `fm/phase-a-hurwitz-decode`, HEAD `f2eba10425118e4ab958aa1170859209f519d0dc`, tree `e337b72cc8dcc0187223f4558f1a0622b83d7422`, 1,493 tracked files.
- Crew split remains unchanged: `phase-a-hurwitz-decode-p1` is the exclusive scheduler/submission/event-source owner; `phase-a-retry3-validation` independently owns audit and result validation and does not submit or poll.

Capsule and one-shot guards:

- Fresh capsule: `/data/scratch-fast/kwen1/compute-native-vq/run-source/phase-a-hurwitz-decode-f2eba1042-retry3.git`; exact HEAD/tree/file count above, bare, shallow, fsck-clean, no alternates, one independent pack with SHA-256 `5a0111f70eb59ec3455cb2ffcac41d1be440a83247f0512fc280f5565e5224fd` and link count 1.
- Launcher: `/data/scratch-fast/kwen1/compute-native-vq/run-state/phase-a-retry3-f2eba1042.sh`, SHA-256 `c4d07f9bb442167b49c52555fe4f5bce327dcaf03a3e872d4ab079008713135c`.
- Materializer: `phase-a-retry3-f2eba1042-materialize.sh` inside the capsule, SHA-256 `1f05983026076f7d8f3a3a3ce774d4c7739fe664105cd357ffa3047353fae003`; it contains the sole `sbatch` command.
- Durable ledger: `/data/scratch-fast/kwen1/compute-native-vq/run-state/phase-a-retry3-f2eba1042.ledger`; it records no retry3 queue/accounting job before helper entry or immediately before submission, an immutable launcher lock, a staged-materializer submit latch, and exactly one numeric job ID. Status: `/data/scratch-fast/kwen1/compute-native-vq/run-state/phase-a-retry3-f2eba1042.status`.

Exact staging and submission:

- The authenticated launcher renewed Kerberos/AFS access and invoked `/bin/bash /afs/csail.mit.edu/u/k/kwen1/.codex/skills/research-reproducibility/scripts/stage_and_run.sh --repo-root /data/scratch-fast/kwen1/compute-native-vq/run-source/phase-a-hurwitz-decode-f2eba1042-retry3.git --staging-parent /data/scratch-fast/kwen1/compute-native-vq/staging -- /bin/bash ./phase-a-retry3-f2eba1042-materialize.sh /data/scratch-fast/kwen1/compute-native-vq/triton f2eba10425118e4ab958aa1170859209f519d0dc e337b72cc8dcc0187223f4558f1a0622b83d7422 1493 /data/scratch-fast/kwen1/compute-native-vq/run-state/phase-a-retry3-f2eba1042.ledger 2026-08-24T19:00:00`.
- Fresh helper stage: `/data/scratch-fast/kwen1/compute-native-vq/staging/20260824_192116-f9abbb-f2eba10-code`.
- Full clean materialized worktree: `/data/scratch-fast/kwen1/compute-native-vq/staging/20260824_192116-f9abbb-f2eba10-code/worktree`; exact HEAD/tree/1,493-file proof above, no alternates.
- Metadata exists in both the staged capsule and materialized worktree. Materialized `REPRODUCIBILITY_METADATA.json` SHA-256: `adf4f15117cfed812877638f3d91e8c9c2f435321eeab3bb6e27272072d41a5a`.
- Sole submission: `sbatch --parsable --account=vision-torralba-urops-meng --qos=vision-torralba-interactive --partition=vision-torralba-rtx3090 --export=ALL,RESEARCH_REPRO_STAGED_DIR=<fresh-worktree>,SOURCE_REPO=/data/scratch-fast/kwen1/compute-native-vq/triton experiments/phase_a_decode/run_phase_a.sbatch`.
- Slurm job `1570434` was accepted at `2026-08-24T19:21:50-04:00`, initially `PENDING` with reason `None`, requesting one node, 4 CPUs, 16 GiB, and exactly one GPU on the required Torralba account/QoS/partition.
- Slurm stdout/stderr: `/data/scratch-fast/kwen1/compute-native-vq/triton/results/2026-08-20-hurwitz-decode-baseline/slurm-1570434.out`.
- Expected result root: `/data/scratch-fast/kwen1/compute-native-vq/triton/results/2026-08-20-hurwitz-decode-baseline/`; expected run outputs remain `staged_snapshot.txt`, `REPRODUCIBILITY_METADATA.json`, `environment.lock.txt`, `system.txt`, `command.txt`, `run_metadata.json`, `correctness.json`, `tuning.json`, `trial_timings.json`, `timings.csv`, `README.md`, and `jh_speedup.png`.

Event and retry gate:

- The sole terminal-state source is `/home/ubuntu/.treehouse/firstmate-557e63/1/firstmate/state/phase-a-hurwitz-decode-p1.check.sh`, mode `0700`, SHA-256 `00a134e48670bc0e8673dc27406bac43cf142a46ff546ef7b4c13da17012b743`; its mode-`0600` `fm-custom-check-v1` trust file binds the same hash. It reads only job `1570434` and atomically wakes once on a terminal Slurm state.
- Prior jobs `1517414` and `1524492` and all prior stages/evidence remain preserved.
- **No fourth experiment submission is authorized.** If job `1570434` is cancelled by UID 0 or otherwise fails, do not submit automatically.

## 2026-08-24 — Retry 3 terminal scheduler evidence

Terminal accounting, captured once at `2026-08-24T19:27:41-04:00`:

- Job `1570434` allocation: `CANCELLED by 0`, reason `None`, exit `0:0`, derived exit `0:0`; admin comment, system comment, and user comment are blank.
- Submitted `2026-08-24T19:21:50-04:00`, eligible at the same time, started `19:21:52`, ended `19:21:53`, elapsed `00:00:01`, node `torralba-3090-1`.
- Allocation: account `vision-torralba-urops-meng`, QoS `vision-torralba-interactive`, partition `vision-torralba-rtx3090`, 4 CPUs, 16 GiB, and one GPU.
- Batch step `1570434.batch`: `CANCELLED`, start/end `19:21:52`/`19:21:53`, elapsed one second. Extern step `1570434.extern`: `COMPLETED`, exit `0:0`, same timestamps and node.
- Retained submit line: `sbatch --parsable --account=vision-torralba-urops-meng --qos=vision-torralba-interactive --partition=vision-torralba-rtx3090 --export=ALL,RESEARCH_REPRO_STAGED_DIR=/data/scratch-fast/kwen1/compute-native-vq/staging/20260824_192116-f9abbb-f2eba10-code/worktree,SOURCE_REPO=/data/scratch-fast/kwen1/compute-native-vq/triton experiments/phase_a_decode/run_phase_a.sbatch`.
- `scontrol show job -dd 1570434` reports `Invalid job id specified`; the terminal record is retained in `sacct`.

Output existence and execution boundary:

- Declared combined stdout/stderr `/data/scratch-fast/kwen1/compute-native-vq/triton/results/2026-08-20-hurwitz-decode-baseline/slurm-1570434.out` is absent.
- The result-root `staged_snapshot.txt`, `REPRODUCIBILITY_METADATA.json`, `environment.lock.txt`, `system.txt`, `command.txt`, `run_metadata.json`, `correctness.json`, `tuning.json`, `trial_timings.json`, `timings.csv`, and `jh_speedup.png` are absent. The only `README.md` there predates this job (`2026-08-21T00:23:22-04:00`). No result-root file changed after submission.
- The batch script writes `staged_snapshot.txt` and copied metadata before environment/hardware capture and invokes `benchmark.py` only afterward. Because none of those pre-benchmark writes or the Slurm log exists, the batch payload did not reach benchmark execution.
- **No scientific result.** Correctness, stability, latency, throughput, and GO/OPTIMIZE-ONCE/KILL remain unmeasured; this terminal state is scheduler evidence only.

Preserved reproducibility evidence:

- Fresh staged capsule `/data/scratch-fast/kwen1/compute-native-vq/staging/20260824_192116-f9abbb-f2eba10-code` and worktree `/data/scratch-fast/kwen1/compute-native-vq/staging/20260824_192116-f9abbb-f2eba10-code/worktree` remain clean at HEAD `f2eba10425118e4ab958aa1170859209f519d0dc`, tree `e337b72cc8dcc0187223f4558f1a0622b83d7422`, with 1,493 tracked files and no alternates.
- Both staged and worktree `REPRODUCIBILITY_METADATA.json` files remain present with SHA-256 `adf4f15117cfed812877638f3d91e8c9c2f435321eeab3bb6e27272072d41a5a`.
- One-shot ledger `/data/scratch-fast/kwen1/compute-native-vq/run-state/phase-a-retry3-f2eba1042.ledger` remains mode `0600`, SHA-256 `b32a3f4cbca1d40f7296e16580cbfb98cc36ff84e4f9992b1b64019d3ed373ac`, and records no preexisting retry3 job, one submit latch, and sole numeric job `1570434`.
- The custom terminal source fired once and is retired after this evidence commit; its exact empty fired marker is removed with it.
- **MANDATORY STOP — NO FOURTH ATTEMPT.** The captain-authorized single retry is consumed. Do not submit, restage, requeue, or cancel another job without a new captain decision.

## 2026-08-25 — Independent CPU-correctness lane recovery and milestone

Recovered provenance and isolation:

- Administrative launch worktree: `/home/ubuntu/.treehouse/triton-ff92c5/3/triton`; detached and clean before creating administrative-only branch `fm/vq-cpu-correctness-r1`. No durable source or notebook content was retained there.
- Clean source checkout checked read-only at `/data/scratch-fast/kwen1/compute-native-vq/triton`; it was on `fm/phase-a-hurwitz-decode`, clean, and ahead of its remote by 12 commits. That existing branch and all prior stages/results/evidence were left unchanged.
- Exact clean provenance/base: `f33a9c88651a1defe2b0a7ef4f80a3cfa1a8f25b` (`Record retry 3 terminal cancellation`).
- Isolated worktree: `/data/scratch-fast/kwen1/compute-native-vq/worktrees/vq-cpu-correctness-r1`; branch: `fm/phase-a-cpu-correctness`; initial HEAD exactly matched the base above.
- Creation command: `git worktree add -b fm/phase-a-cpu-correctness /data/scratch-fast/kwen1/compute-native-vq/worktrees/vq-cpu-correctness-r1 f33a9c88651a1defe2b0a7ef4f80a3cfa1a8f25b`.
- The first inline patch transport failed twice before writing any substantive file (one local command-construction failure and one rejected corrupt patch); the scratch worktree remained clean. Recovery used mode-`0600` temporary files copied with `scp`, matching SHA-256 hashes before and after transfer, and a same-filesystem atomic directory move. The temporary CSAIL transfer directory and every VM-local transfer artifact were removed after verification; the administrative worktree was then clean.

Current diagnostic tuple evidence (read-only carry-forward):

- Preserved diagnostic GPU smoke job `1574206` demonstrates a current success for account/QoS/partition tuple `vision-torralba-urops-meng` / `vision-torralba-interactive` / `vision-torralba-rtx3090` on `torralba-3090-1.csail.mit.edu`. Its output records `slurm_job_id=1574206`, one visible RTX 3090, and return code `0` for both diagnostic `nvidia-smi` calls.
- Report/one-shot submission ledger: `/data/scratch-fast/kwen1/torralba-gpu-smoke-r1-corr-3c72e85347313df9/submission-ledger.txt`, SHA-256 `381a14fd1e6e4dc7b4d764783e303839f94cbb652cdedc6577076655deb1f8b9`.
- Diagnostic output: `/data/scratch-fast/kwen1/torralba-gpu-smoke-r1-corr-3c72e85347313df9/1574206.out`, SHA-256 `b9ac17b5ad8d89c243769699380a6be5cff6f09a527a34543a5c79fcc79e88ac`.
- This evidence was only read from preserved files. This lane ran no Slurm command, scheduler poll, allocation command, GPU query, CUDA/Triton kernel, or experiment. The smoke validates the current resource tuple only; it is not a Phase A decoder result and does not supersede the UID-0 cancellation evidence for jobs `1517414`, `1524492`, and `1570434`.

CPU-correctness implementation:

- Exclusive source root: `experiments/phase_a_decode/cpu_correctness/`; no edit was made to `benchmark.py`, `run_phase_a.sbatch`, existing result directories, or another worker's directory.
- `oracle.py` implements the independent scalar-first `(w,x,y,z)` PyTorch contract, the 24 ordered Hurwitz primary units, `id = p*S+s`, J joint-table gathering, F generic Hamilton multiplication, and H signed-permutation/half-unit specialization. All public tensor operations fail closed on non-CPU devices.
- `test_cpu_correctness.py` exhausts all 24 primary units and all secondary indices for representative `S = 1, 2, 7, 31`; checks scalar-first basis orientation and flat-ID decomposition; covers the 8 signed permutations, all 16 half-unit sign masks, zero/signed-zero, basis, mixed-sign, and power-of-two edges; validates fp16 and bf16 storage round trips; and exercises deterministic property cases with exact seeds `0`, `20260824`, and `0xC0FFEE`.
- `README.md` documents the no-GPU boundary and integration seam. `benchmark.py` is intentionally not imported because it couples the contract to top-level Triton/plotting dependencies and a CUDA execution path; a future GPU-only adapter can compare copied-back kernel outputs against `decode_variants()` without weakening this lane.
- Transferred source hashes: `README.md` `b353bc23ded23cc285124ffdee10dd738e2b110f52d0e08c6f9a23da98bf4b57`; `__init__.py` `de1c288d069c711696cd7c06656c6b9f6bedd151a77b7efc9712996b12dba906`; `oracle.py` `564db778450a4441f2e85aa786e7100ba997dc7def096e16f0e3c2d0f65b6da9`; `test_cpu_correctness.py` `d26029e761401184fa12b555aed42fdfb72b12c7ca806e6d9253a535944f4d9a`.

CPU-only validation and outcome:

- Host: `slurm-login-0.csail.mit.edu`. Command environment: `CUDA_VISIBLE_DEVICES=""`; Python `3.10.20` (conda-forge, GCC 14.3.0); PyTorch `2.8.0+cu128`; `torch.cuda.is_available()` was `False`; default dtype was `torch.float32`.
- The system `python3` lacked PyTorch. The scratch environment `/data/scratch-fast/kwen1/micromamba/root/envs/causal_forcing/bin/python` supplied PyTorch but not pytest, so the dependency-free standard-library unittest runner was used instead of modifying a shared environment.
- Exact test command: `CUDA_VISIBLE_DEVICES="" /data/scratch-fast/kwen1/micromamba/root/envs/causal_forcing/bin/python -m unittest discover -s experiments/phase_a_decode/cpu_correctness -p "test_*.py" -v`.
- Result: `Ran 10 tests in 0.345s` and `OK`; zero test failures, errors, skips, or unsupported dtype cases. Both CPU fp16 and CPU bf16 round-trip checks passed. No result directory or experiment artifact was created.
- Static checks: `git diff --check` passed; `CUDA_VISIBLE_DEVICES="" .../python -m compileall -q experiments/phase_a_decode/cpu_correctness` passed. Generated `__pycache__` was removed before commit. Per repository guidance, `make` was not run because changes are Python-only.

Decision boundary:

- The independent CPU mathematical contract is validated and ready for direct PR review. This is a correctness-test milestone only: no latency, throughput, or device-kernel conclusion follows.
- **PHASE A REMAINS CLOSED.** Preserve the UID-0 blocker and the no-fourth-attempt gate. This lane makes no GO, KILL, pivot, submission, retry, or performance claim.

## 2026-08-25 — Phase A submission-free benchmark preflight (`fm/phase-a-benchmark-preflight`)

Provenance and isolation:

- Administrative launch worktree: `/home/ubuntu/.treehouse/triton-ff92c5/4/triton`; `pwd -P` and `git rev-parse --show-toplevel` both resolved exactly to it before the administrative branch `fm/vq-benchmark-preflight-r1` was created. No substantive file or commit was made there.
- Scratch source: `/data/scratch-fast/kwen1/compute-native-vq/triton`; clean branch `fm/phase-a-hurwitz-decode`, exact HEAD/base `f33a9c88651a1defe2b0a7ef4f80a3cfa1a8f25b`.
- Isolated scratch worktree: `/data/scratch-fast/kwen1/compute-native-vq/worktrees/vq-benchmark-preflight-r1`; branch `fm/phase-a-benchmark-preflight`, created directly from `f33a9c88651a1defe2b0a7ef4f80a3cfa1a8f25b` only after the source was verified clean and both the branch and target path were verified absent.
- Frozen hashes: `benchmark.py` SHA-256 `c2c451ab3290af8d4fc97ff8bec077533a679ffd825a2d44d8907dc8cdd16a6a`; `run_phase_a.sbatch` SHA-256 `aca356cc4d1ba87ea67c94f6bca671e52e0a26b74a779722fae218dd01ba8bc6`. Neither file was edited.

Exact scratch setup command (after read-only guards):

```text
git -C /data/scratch-fast/kwen1/compute-native-vq/triton worktree add -b fm/phase-a-benchmark-preflight /data/scratch-fast/kwen1/compute-native-vq/worktrees/vq-benchmark-preflight-r1 f33a9c88651a1defe2b0a7ef4f80a3cfa1a8f25b
```

Implementation and dry validation:

- Added only `experiments/phase_a_decode/preflight/`: a pinned `contract.json`, standard-library-only static validator, labeled safe fixtures, and usage/admin-gate documentation.
- Exact development live command: `python3 experiments/phase_a_decode/preflight/validate_preflight.py --allow-owned-dirty` → `PASS live: frozen Phase A contract is internally consistent and submission-free`; the override accepts dirt only within this lane's two owned paths, and the committed clean gate is recorded separately below.
- Exact fixture command: `python3 experiments/phase_a_decode/preflight/validate_preflight.py --fixtures-dir experiments/phase_a_decode/preflight/fixtures` → exit `0`.
- `valid-full-stage-metadata-and-contract` → expected pass, passed.
- `reject-missing-full-stage-metadata` → expected fail, rejected for missing `REPRODUCIBILITY_METADATA.json`.
- `reject-compact-capsule-instead-of-full-source` → expected fail, rejected for source mismatch.
- `reject-stage-equal-to-source` → expected fail, rejected for source/stage equality and invalid stage parent.
- `reject-non-torralba-account` → expected fail, rejected for account drift.
- `reject-relative-result-directory` → expected fail, rejected for nonexact/nonabsolute results.
- `reject-masked-critical-exit` → expected fail, rejected for masked critical exit.
- Additional CPU-only checks: `python3 -m py_compile experiments/phase_a_decode/preflight/validate_preflight.py`, `python3 -m json.tool` on the manifest and every fixture, frozen-wrapper `bash -n`, and `git diff --check`; all passed.

Gate and outcome:

- The manifest pins the named branch/base, full-stage metadata fields and exact source identity, source/stage inequality, Torralba-only account `vision-torralba-urops-meng`, QoS `vision-torralba-interactive`, partition `vision-torralba-rtx3090`, one node/task/GPU, 4 CPUs, 16 GiB, one hour, absolute result/log paths, all 13 expected artifacts including the Slurm log, and fail-closed exit propagation.
- Smallest administrative evidence before a future retry is scientifically justified: one administrator-authenticated controller/daemon diagnosis tied to job `1570434` that names the cancellation cause and either confirms it was cleared or supplies the exact permitted Torralba invocation. A new explicit fourth-submission authorization is separately required.
- This lane does not claim the UID-0 cause is fixed. It contacted no scheduler, created no full or partial stage, ran no GPU/benchmark code, changed no results/evidence/retry ledger, and created no experiment stage, Slurm job ID, or event source.

## 2026-08-25 — Append-only notebook reconciliation for preflight reuse

- Cherry-pick `324f397bbaa8d0045f5a556da9ffd19edf87100c` conflicted only at the append point in `notebook.md`. Resolution retained the complete CPU-correctness lineage followed by the complete submission-free preflight lineage; no historical entry was deleted or rewritten.

- Clean committed gate: at commit `324f397bbaa8d0045f5a556da9ffd19edf87100c`, exact command `PYTHONDONTWRITEBYTECODE=1 python3 experiments/phase_a_decode/preflight/validate_preflight.py` returned `PASS live: frozen Phase A contract is internally consistent and submission-free` with the worktree clean.

## 2026-08-25 — Append-only notebook reconciliation for clean preflight validation

- Cherry-pick `d96a5a710309b11f8a7eefbdf11d0fc28403404f` conflicted only because the local lineage had already appended CPU-correctness, preflight, and reconciliation entries. Resolution retained every local entry and appended the commit's complete clean-gate entry; no historical entry was deleted or rewritten.

## 2026-08-25 — Phase A GPU correctness-only r1 pre-launch record

Scientific scope and distinction:

- Fresh captain authority permits exactly one bounded GPU correctness-only experiment. Unlike cancelled jobs `1517414`, `1524492`, and `1570434`, this payload stops immediately after the existing exhaustive/random J/F/H checks: it does not call tuning, timing, CUDA graphs, latency trials, `decide`, or any GO/OPTIMIZE/KILL gate.
- Coverage is fixed at `S=96,192`, fp16 plus bf16 when supported, 24 primary units and every feasible secondary index in each exhaustive case, the existing random cases, and configs `b256-w4` and `b512-w8`. Required checks are finite values, J bitwise gather equality, F/H axis bitwise equality to J, and F/H tolerances against the independent scalar-first PyTorch oracle.
- Correctness PASS can only unlock a separately controlled timing decision; this lane makes no broader Phase A/B timing or Gate B claim. Independent audit ownership remains `vq-phase-a-gpu-correctness-audit-r1`, which neither submits nor polls.

Source and code provenance before this append:

- Source worktree `/data/scratch-fast/kwen1/compute-native-vq/worktrees/vq-phase-a-gpu-correctness-r1`, branch `fm/phase-a-gpu-correctness-r1`, clean implementation commit `28d1b5e968af17f5829ef9e270945c5665763769`, tree `aeca910a220e4c809db36fe89c8540ddd56ea2c3`. This notebook-only append will be committed before staging; the exact sealed launch commit/tree and clean local state will be captured in `REPRODUCIBILITY_METADATA.json`, stage proof, and the post-submission entry.
- `benchmark.py` SHA-256 `4d47e2575b4b086237f8971b99d204c4315b8d09d024ab8b27fc40ba16137606`.
- GPU runner README SHA-256 `55cb29c78a2385bb0fa7662699310a055b9fe35d020c3156b55fb879077b1dd8`.
- `run_gpu_correctness.sbatch` SHA-256 `81448a6b679026827b044146d27adbd9a50f4ea0973a2f4d645bbd0dbf846f76`.
- `submit_from_stage.py` SHA-256 `b539ecf03554e3ffae268b3f3d0b83b7c3bb35478c37991b1e69c05726bcd623`.
- `validate_environment.py` SHA-256 `c9dc5e5d6813aa1eac73dbc75cab7436fc12326277faacd677c8b61b0328b689`.
- Reused preflight contract SHA-256 `cd10962b4f70e73493a7175eae7dafe16ec01829d92fdcddadc39e12f2b45c57`.

Launch contract:

- Fresh result root: `/data/scratch-fast/kwen1/compute-native-vq/results/2026-08-25-vq-phase-a-gpu-correctness-r1`; it contained only the scheduler preflight record before staging.
- Bounded live evidence at `2026-08-25T16:32:40-04:00` confirms user `kwen1` is associated with account `vision-torralba-urops-meng`, that the association includes QoS `vision-torralba-interactive`, and that partition `vision-torralba-rtx3090` is UP and allows both. Evidence: `scheduler_preflight.txt`, SHA-256 `3d8de3ddb503e24944af660d7d6534cc964f596d8c096f731bb6e66c72df70a3`.
- Intended maximum allocation: one Torralba node, one task, one RTX 3090 GPU, 4 CPUs, 16 GiB, and 15 minutes, using the exact live account/QoS/partition above. There will be exactly one `sbatch` attempt, with no interactive allocation, retry, requeue, or cancellation.
- Immediately before that sole attempt, invoke `/afs/csail.mit.edu/u/k/kwen1/.codex/skills/research-reproducibility/scripts/stage_and_run.sh` from the clean source with staging parent `/data/scratch-fast/kwen1/compute-native-vq/staging`. The staged `submit_from_stage.py` must verify `.git`, metadata, clean source identity, tracked path set/content, commit/tree, and local state before its single fixed `sbatch` call.
- The allocated-node payload must write validation and final manifests even on failure and must exit before CUDA correctness if allocation, GPU, CUDA/PyTorch, VRAM, host memory, or source/stage/result storage validation fails.

## 2026-08-25 — Phase A GPU correctness-only r1 full-stage transport failure

Preserved launch boundary:

- Sealed clean source immediately before the helper: branch `fm/phase-a-gpu-correctness-r1`, commit `3c7614c1024ebf420426da1b4c712e71ac95f0f3`, tree `ae70f01148bb6a5afe2ed17059fa277404ec3789`.
- Exact foreground command invoked the required AFS `stage_and_run.sh` with source `/data/scratch-fast/kwen1/compute-native-vq/worktrees/vq-phase-a-gpu-correctness-r1`, staging parent `/data/scratch-fast/kwen1/compute-native-vq/staging`, and staged command `python3 experiments/phase_a_decode/gpu_correctness/submit_from_stage.py --source /data/scratch-fast/kwen1/compute-native-vq/worktrees/vq-phase-a-gpu-correctness-r1 --result /data/scratch-fast/kwen1/compute-native-vq/results/2026-08-25-vq-phase-a-gpu-correctness-r1`.
- The helper announced target `/data/scratch-fast/kwen1/compute-native-vq/staging/20260825_163420-b041a0-3c7614c10-code`, then the tracked SSH transport ended with status `255` after about 54 seconds before `Staging complete`.
- The incomplete 21 MiB stage remains preserved. It has no `REPRODUCIBILITY_METADATA.json`; the staged verification wrapper never ran; `stage_verification.json`, `submission_request.txt`, and `submission_response.json` are absent; and no matching helper, rsync, or submission-wrapper process remained.
- The durable result root contains only the pre-submission `scheduler_preflight.txt`; no Slurm log, environment validation, correctness record, run metadata, or final result manifest exists.

Outcome and boundary:

- **NO SUBMISSION / NO RESULT.** The sole `sbatch` site is inside `submit_from_stage.py`, and the missing staged metadata and wrapper artifacts prove execution did not reach it. No numeric job ID exists, so no terminal-state source was created or registered.
- Per the reproducibility skill's fail-closed staging rule, do not bypass the helper, submit from source, or launch CUDA correctness from this incomplete stage. No retry, requeue, cancellation, or manual scheduler polling was performed.
- Next dependency: supervisor direction for a policy-compliant way to complete the mandatory full-repository helper stage without weakening the one-`sbatch`, staged-only, and no-retry boundaries.

## 2026-08-25 — Phase A GPU correctness-only r1 authorized staging recovery

- Supervisor confirmed from the preserved boundary evidence that the first helper transport ended during copy, before metadata, staged verification, `sbatch`, or any job ID. The one authorized `sbatch` attempt is therefore unused, and exactly one fresh helper staging recovery is authorized without changing experiment scope.
- Recovery source before this append: clean branch `fm/phase-a-gpu-correctness-r1`, commit `d72243dd5549281a1454d6c937b0a92e038f7c41`, tree `e8806a5a972fa33e7b589afe2b3dc7bada2e1a69`. This recovery entry will be committed before launch so the helper sees a clean sealed source.
- Failed stage `/data/scratch-fast/kwen1/compute-native-vq/staging/20260825_163420-b041a0-3c7614c10-code` remains incomplete with no metadata. Its read-only path/type/size/mtime manifest SHA-256 is `782581a29852065d727f666e9be951e5920567cf79276ef3677fd4a05cebdcc3`; it must never be reused, repaired, or deleted.
- Durable result root remains `/data/scratch-fast/kwen1/compute-native-vq/results/2026-08-25-vq-phase-a-gpu-correctness-r1` and still contains only `scheduler_preflight.txt` before recovery.
- The recovery uses the same required AFS `stage_and_run.sh`, source, staging parent, correctness-only code/config hashes, result root, and eligible tuple `vision-torralba-urops-meng` / `vision-torralba-interactive` / `vision-torralba-rtx3090`, bounded to one node/task/GPU, 4 CPUs, 16 GiB, and 15 minutes.
- `sbatch` remains gated on a newly announced timestamped stage reaching `Staging complete`, producing `REPRODUCIBILITY_METADATA.json`, and passing the staged wrapper's exact source/config byte verification. Any fresh transport failure, submission ambiguity, or verification drift requires an immediate stop with no further staging or submission retry.

## 2026-08-25 — Phase A GPU correctness-only r1 sole submission accepted

Fresh stage and immutable proof:

- The foreground PTY-backed recovery invoked the same required AFS helper and reached `Staging complete` for fresh stage `/data/scratch-fast/kwen1/compute-native-vq/staging/20260825_164006-0015a3-9dc0f26d1-code`; the failed stage `20260825_163420-b041a0-3c7614c10-code` remains untouched.
- Staged `REPRODUCIBILITY_METADATA.json` records clean source commit `9dc0f26d19a49f21201d048d28059662b94abc39`, tree `f7a770e5be03dfbdaafea73ce79e3c377603029a`, and exact source/stage paths; SHA-256 `857048b9e389c608e6fdae9f5b0eb5c0866b75d44808f7bc495b107e722e1163`.
- `stage_verification.json` records 1,511 identical tracked paths, aggregate content SHA-256 `7839f4ff799bd31c7936b26dd45085788cb75a63d2f0344a5343dd86fc3275ef`, identical `.git` pointer SHA-256 `62fee656b5ed519398878f4419098431b2f79f8e6f91bc37d004eb1a80480420`, and clean local state; file SHA-256 `1441191eae67ddcca09bd914b9e6af3826aedc5f3043b5caf8ff112f58e32434`.

Sole submission:

- Exact command: `sbatch --parsable --account=vision-torralba-urops-meng --qos=vision-torralba-interactive --partition=vision-torralba-rtx3090 --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --mem=16G --time=00:15:00 --output=/data/scratch-fast/kwen1/compute-native-vq/results/2026-08-25-vq-phase-a-gpu-correctness-r1/slurm-%j.out --error=/data/scratch-fast/kwen1/compute-native-vq/results/2026-08-25-vq-phase-a-gpu-correctness-r1/slurm-%j.out --export=ALL,SOURCE_REPO=/data/scratch-fast/kwen1/compute-native-vq/worktrees/vq-phase-a-gpu-correctness-r1,RESULT_DIR=/data/scratch-fast/kwen1/compute-native-vq/results/2026-08-25-vq-phase-a-gpu-correctness-r1 experiments/phase_a_decode/gpu_correctness/run_gpu_correctness.sbatch`.
- Response: return code `0`, stdout numeric job ID `1579631`, stderr `sbatch: partition vision-torralba-rtx3090, qos vision-torralba-interactive`. Request SHA-256 `eb354289555fb87fca46ee153a2aee992e9def8d93c4f5c3a214e1cdf28971c1`; response SHA-256 `ab01f3f89e69eaed0eba2bb4e76afb8c682edf4a04150524ad378c26d5fe52f6`.
- Durable result root: `/data/scratch-fast/kwen1/compute-native-vq/results/2026-08-25-vq-phase-a-gpu-correctness-r1`; combined stdout/stderr: `slurm-1579631.out`. The staged payload alone owns environment validation, correctness, and final result manifests.

Exclusive terminal source:

- Sole monitoring owner is task `vq-phase-a-gpu-correctness-r1` through fixed-host/fixed-job check `/home/ubuntu/.treehouse/firstmate-557e63/1/firstmate/state/vq-phase-a-gpu-correctness-r1.check.sh`, mode `0700`, SHA-256 `997b647242bd71eec798fc7b8616751cd89bd82a1d018ff95f86cf85e3514ca0`, bound by its mode-`0600` `fm-custom-check-v1` trust record.
- The check queries only `sacct -X` for job `1579631`, stays silent for nonterminal states and transient errors, and atomically emits once for a terminal state. No conversational/manual scheduler polling, retry, requeue, or cancellation is permitted.

## 2026-09-02 — Phase A clean correctness rerun preparation

Current state:

- Preparation only on branch `fm/cnvq-clean-rerun-prep-r1`. No stage, result, scheduler
  attempt, GPU work, timing, event source, push, PR, or merge was created.
- The authoritative plan, direct handoff, and independent audit were read in full. Their
  SHA-256 values are `9fffe0a4a46ed88f10a7ad2961cad01bb9defdd89fd80b96a08bb7d6c17681ee`,
  `37bc4521665b8c2aef98fa71d82163f5443f7c3503f6e13afd88a721cb673852`, and
  `3e3d91c42d36301f0f5542216ed355d7e30aa32ca8178f03da2835a62007940e`.
- Read-only Git upload-pack retrieved exact commits `e2161ae54d3b215c768b0f30584cb099b19102a3`
  and `9dc0f26d19a49f21201d048d28059662b94abc39` from the handed scratch worktree.
  Both commit objects rehashed to their stated IDs. Before preparation edits, the recovered
  `experiments/phase_a_decode` tree matched `e2161ae5...` exactly at tree
  `2e158d205fe2c7f72768c0565e8a28f953db490e`.

Preparation changes:

- `gpu_correctness/prepare_clean_rerun.py` creates a detached standalone clone with
  `--no-local --no-hardlinks`, rejects external Git/object dependencies, runs the CPU
  preflight, audits the helper-produced full stage, freezes it read-only, and binds a full
  recursive inventory into an immutable launch manifest.
- The manifest digest binds the submission request, exported environment, first runtime
  artifact, run metadata, final manifest, and fsynced hash-chained attempt ledger. An
  exclusive launch lock allows one invocation; accepted, failed, and ambiguous responses
  cannot be retried. Raw response bytes are preserved.
- Runtime classification now reserves scientific `FAIL` for named finite, bitwise, or fixed
  tolerance comparisons. Environment, allocation, compilation, launch, malformed or
  incomplete artifacts, and post-write failures are `NO RESULT`. A post-write fault keeps a
  completed comparison `PASS` separate from overall run completion.
- The fixed correctness contract remains `S={96,192}`, seed 0, two roles, eight KV heads,
  head dimension 128, exhaustive/random inputs, and configs `b256-w4`/`b512-w8`. It requires
  eight fp16 records plus eight bf16 records when supported. Fp16 thresholds remain
  `max_abs <= 4e-3` and `relative_fro <= 1e-3`; the declared bf16 direct-float32 asymmetry is
  unchanged. Timing paths and artifacts are rejected.

Validation and remaining gate:

- The standard-library behavioral suite covers external `.git` rejection, standalone clone
  ownership, read-only stage inventory, stale scheduler evidence, manifest/request binding,
  success and failure one-shot ledgers, scientific/infrastructure/post-write classification,
  exact record counts, workload drift, and timing rejection.
- Local Python has neither PyTorch nor pytest. The independent PyTorch CPU oracle therefore
  remains a launch-time preflight gate. It must pass from the detached standalone clone at
  final commit `R` before any stage is created.
- `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s
  experiments/phase_a_decode/gpu_correctness -p 'test_clean_rerun.py' -v` passed all 18
  preparation/protocol/classification tests. `shellcheck` and `bash -n` passed the job
  wrapper, all seven Phase A Python entrypoints compiled from source bytes, the protocol
  CLI exited zero, and `git diff --check` passed. No build was run because all preparation
  changes are Python, shell, and documentation only.
- The corresponding local CPU-oracle command stopped at import with
  `ModuleNotFoundError: No module named 'torch'`; `python3 -m pytest` is likewise unavailable.
  Neither missing local dependency is converted into a preflight PASS.
- A future authorized owner must capture a fresh eligible Torralba tuple, finalize the
  scheduler contract, run the research staging helper from the standalone clone, and inspect
  the prepared manifest/request/ledger before deciding whether to invoke the one-shot submit
  command. A new accepted job would require its own single event source.

## 2026-09-03 — Phase A clean correctness rerun authorization

- The captain authorized exactly one fresh Phase A correctness-only run from preparation
  commit `337f6a738ad069c443b71e761d2c96ef7d20c22f`. The correctness workload, thresholds,
  record counts, failure taxonomy, immutable-stage gate, and prohibition on timing, tuning,
  Phase B, retry, requeue, and cancellation remain unchanged.
- Before staging, the one-shot submitter was corrected to follow the TIG environment rule:
  the immutable request omits every `sbatch --export` option and supplies its manifest-bound
  variables in the `sbatch` process environment, allowing Slurm's safe default `ALL` export.
- A new final source commit and tree will be detached into a standalone scratch clone. A
  fresh CPU preflight, live Torralba selection, complete stage audit, immutable launch
  manifest, and registered task-owned terminal checker remain mandatory before the sole
  submission.

## 2026-09-03 — Phase A stage-only continuation correction

Current state:

- The task-owned progress wrapper let the required helper finish `--stage-only` in 54.37
  seconds at fresh stage `20260903_141105-83ea60-0d79ada-code`; its SHA-256 remained
  `8f98d17eafac255f294735aba35dfe97657b56bb9757ce1afee4ef6591225caf`.
- Read-only verification matched 2,193 source/stage paths and 37,662,404 content bytes at
  commit `0d79adad95c5d699fb21c9bd5d9aa8965f7fa211`, tree
  `7df1e089186a93615a7e9cf167a3de254f7fa5b3`, with a clean tracked state, owned `.git`,
  full `git fsck`, and no alternates.
- The separate verifier correctly stopped because stage-only helper metadata truthfully has
  an empty command while the old gate accepted only an in-helper `freeze-stage` command.
  No checker, `sbatch`, GPU work, timing, retry, repair, or access to job `1579631` occurred.

Correction:

- `freeze-stage --stage-only-continuation` accepts only that exact empty helper command and
  requires an immutable `stage_only_attestation.json`, and rechecks every source, stage,
  Git, metadata, scheduler, tracked-content, and complete-tree byte gate. The attestation
  binds the exact required helper path and approved SHA-256, wrapper path and hash, complete
  helper log and hash, foreground timing, zero exit, empty helper command, and fresh stage
  identity. Symlinks, partial copies, mismatched invocation state, and nonempty commands are
  rejected.
- Before freezing, continuation exclusively creates mode-`0444` `stage_verification.json`
  inside the stage. It records the helper attestation, metadata, complete-tree digest, and
  exact separate Python argv, working directory, and helper-equivalent environment; the
  launch manifest binds both hashes. The stage-local record prevents a second result root
  from rebinding the same stage, including after an interrupted post-binding preparation.
- The submitter rejects every ambient `SBATCH_*` variable before creating its launch lock
  or appending the ledger. Scheduler resource counts, memory, wall time, and minimum VRAM
  accept only JSON integers, never booleans or floating-point values.
- Allocated-node validation recognizes the stage-local binding only when the continuation
  manifest names `stage_verification.json` and its SHA-256 matches. The file remains covered
  by the full immutable stage inventory; any absent, renamed, symlinked, or changed binding
  is `NO RESULT` before CUDA.
- Local validation passed 28 behavioral preparation/protocol/classification tests,
  `shellcheck` and `bash -n` for both Phase A batch scripts, source-byte compilation of all
  11 Phase A Python files, the correctness-contract CLI, the preparation CLI, and
  `git diff --check`. No native build was needed because this correction changes only
  Python and documentation.
- The correctness workload, record counts, fp16 thresholds, bf16 policy, no-timing scope,
  one-submission ledger, TIG default-`ALL` export behavior, and Phase B prohibition are
  unchanged.

## 2026-09-03 — Bytecode-free source verification

Current state:

- The resumable-source comparison found only three ignored verifier bytecode files under
  `experiments/phase_a_decode/gpu_correctness/__pycache__`. Their embedded source paths
  made the otherwise complete trees differ by three bytes.
- The preparer now disables bytecode writes before importing repo-local modules. Every
  launch-time preparer and inventory process also runs with `PYTHONDONTWRITEBYTECODE=1`
  and Python `-B`.
- A subprocess regression removes ambient bytecode suppression, invokes the preparer
  entrypoint, and requires its complete source inventory to remain unchanged.
- Local validation passed 29 behavioral preparation/protocol/classification tests,
  `shellcheck` and `bash -n` for the Phase A batch script, source-byte compilation of all
  11 Phase A Python files, both contract/preparer CLIs, and `git diff --check`.
- The fixed workload, thresholds, record counts, bf16 policy, correctness-only scope,
  single-submit policy, and Phase B prohibition remain unchanged.

Next step:

- Commit the verifier fix, refresh and hash the resumable scratch source to that exact
  commit, then create one new helper-owned immutable stage. No submission occurs before
  the stage and task-owned terminal checker both pass.
