# Research notebook

## 2026-08-30 — Structured-rotation secondcrew GPU-visibility smoke r2

This lane is authorized for exactly one driver-only CSAIL visibility smoke on
branch `fm/structured-rotation-secondcrew-gpu-smoke-r2`, based on
`f893845b9b91599ebd3b7a9c7f28164f39c7ed94`. It makes no scientific or
performance claim and does not import Python, Torch, Triton, model code, or a
scientific payload.

Read-only scheduler preflight at `2026-08-30T23:11:25Z` used the task's
local-tmux-owned SSH route to `kwen1@slurm-login.csail.mit.edu`. The account
association included `vision-torralba-urops-meng` with
`vision-torralba-interactive`. The Torralba GPU snapshot showed the dedicated
H100 partition allocated, H200 mixed, RTX 3090 mixed, and both V100 nodes idle.
The architecture-agnostic payload therefore selected the smallest adequate
idle option, `vision-torralba-v100`, rather than preferring a larger GPU.

The exact authorized request is one `salloc` for one node/task/GPU, two CPUs,
`8G`, and ten minutes, followed after grant by one `srun --pty` invocation of
`scripts/csail_gpu_visibility_smoke.sh` from a fresh independent full-repository
stage under `/data/scratch-fast/kwen1`. The stage, result paths, manifests,
hashes, one-shot latch/ledger, exact quoted argv, allocation evidence, and
single process-event-source lifecycle are recorded below after they become
durable. No allocation has been requested at this preparation point.
