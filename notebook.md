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
