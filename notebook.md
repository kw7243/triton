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

## 2026-08-25 — Phase A analysis schema and synthetic visualization lane

Provenance and isolation:

- Administrative launch checkout: `/home/ubuntu/.treehouse/triton-ff92c5/5/triton`, verified as the disposable top level before creating bookkeeping branch `fm/vq-analysis-viz-r1`; no substantive file was written there.
- Source checkout: `/data/scratch-fast/kwen1/compute-native-vq/triton`; it was clean at exact HEAD `f33a9c88651a1defe2b0a7ef4f80a3cfa1a8f25b` before worktree creation.
- Base commit: `f33a9c88651a1defe2b0a7ef4f80a3cfa1a8f25b` (`Record retry 3 terminal cancellation`).
- Analysis branch: `fm/phase-a-analysis-viz`.
- Isolated remote worktree: `/data/scratch-fast/kwen1/compute-native-vq/worktrees/vq-analysis-viz-r1`, created directly from the base commit and clean before edits.
- Origin: `https://github.com/kw7243/triton.git`.
- Exclusive implementation scope: `experiments/phase_a_decode/analysis/`, `results/2026-08-24-phase-a-analysis-synthetic/`, and this uniquely titled append-only notebook section.
- `benchmark.py`, `run_phase_a.sbatch`, the real baseline result root, prior stages, and all evidence files were left unchanged.

Fixture and schema:

- Executable schema version: `phase-a-analysis-v1`.
- Deterministic fixture seed: `20260824`.
- Fixture input root: `results/2026-08-24-phase-a-analysis-synthetic/inputs/`.
- The fixture metadata declares `synthetic=true`, the seed/schema above, base commit and analysis branch, and a synthetic device string. The fixture was created without CUDA, a GPU, Slurm, polling, staging, submission, a job ID, or an event source.
- The validator requires `correctness.json`, `tuning.json`, `trial_timings.json`, `timings.csv`, and `run_metadata.json`; it rejects missing/malformed/duplicate/partial data and recomputes quantiles, medians, stability, throughput, J/H ratios, configuration selections, and cross-file trial/timing agreement.
- Aggregation reports descriptive J/H and F/H hot/cold comparisons plus J/F/H stability summaries. It does not call the benchmark decision logic and produces no scientific gate or weight-VQ recommendation.
- Every generated CSV row, the Markdown heading/footnote, and the SVG metadata/visible watermark say `SYNTHETIC — NOT A SCIENTIFIC RESULT`.

Exact generation commands, run from the isolated remote worktree:

```bash
python3 -m experiments.phase_a_decode.analysis.make_synthetic_fixture --output results/2026-08-24-phase-a-analysis-synthetic/inputs
python3 -m experiments.phase_a_decode.analysis.analyze --input results/2026-08-24-phase-a-analysis-synthetic/inputs --output results/2026-08-24-phase-a-analysis-synthetic
```

Exact validation commands and outcomes:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s experiments/phase_a_decode/analysis/tests -v
# 9 tests passed: complete/deterministic output, missing, malformed, duplicate JSON key,
# duplicate timing cell, unstable, partial, and cross-file mismatch coverage.

PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
import xml.etree.ElementTree as ET
from pathlib import Path
ET.parse(Path("results/2026-08-24-phase-a-analysis-synthetic/comparison.svg"))
print("comparison.svg: valid XML")
PY
# comparison.svg: valid XML

git diff --check
# passed with no output
```

- Protected-scope check found no changes to `experiments/phase_a_decode/benchmark.py`, `experiments/phase_a_decode/run_phase_a.sbatch`, or `results/2026-08-20-hurwitz-decode-baseline/`.
- `make` was intentionally not run because this lane changes only Python/docs/data and no native/compiler code.
- Versions: Python `3.10.12`; git `2.55.0`; Linux `5.15.0-190-generic #200-Ubuntu SMP Fri Aug 7 15:06:04 UTC 2026 x86_64 GNU/Linux`.
- Generated artifacts: `analysis_table.csv` SHA-256 `4ff417f6ce4e9ef2e155b0e5a9dd242a9aa230bf3584387395f8dc241d56ca42`; `analysis_table.md` `ab1c496516d3dfac731067113d3e06cac9a21efb3fd13345e673cdcfb11d00dd`; `comparison.svg` `1d92347c9e4c5f5b266391c0c4b53c77448756120e22eda310b0f4b53aafbf64`; `validation_report.json` `9134329ca2f205dd1cc2319a213396c9a625502cd40667a9b3a1003a5e0da365`.
- `validation_report.json` records validation `passed`, six timing cells, eight correctness cases, zero unstable synthetic cells, all five input SHA-256 values, and `scientific_decision: null`.
