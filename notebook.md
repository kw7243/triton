# Research notebook

## 2026-08-31 — Structured-rotation secondcrew GPU-visibility smoke r3

Authorized scope is exactly one corrected, fresh, driver-only CSAIL
GPU-visibility smoke on branch
`fm/structured-rotation-secondcrew-gpu-smoke-r3`, with at most one `salloc`
and one `srun --pty`. The run deliberately excludes Python, Torch, Triton,
models, benchmarks, and every scientific or performance claim.

Gate 1 diagnosed r2 before any scheduler mutation. The same valid r2 bundle
failed with `error: need a repository to verify a bundle` outside a Git
repository and succeeded inside an initialized repository and inside a proven
ordinary independent clone. An invalid bundle truncated inside its header
failed from that same repository, so cwd alone is not sufficient for arbitrary
bytes. The r2 correction used in-place redirection to its already mode-`0500`
helper and failed before replacing the stale bytes; a fresh mechanics test
proved that a verified sibling temporary file can instead be atomically renamed
over that target in a writable owned parent. The bounded diagnosis is retained
at `/data/scratch-fast/kwen1/structured-rotation-secondcrew-gpu-smoke-r3/attempt-20260831T013733Z-f893845b9b91/preflight/diagnosis/diagnosis.md`
(SHA-256
`d742d15ff88c6bdb569de3a1de2fc9380abc3010c393d7d179822bac33874cc0`).
No r2 evidence was modified or executed, and no Slurm request has been made.

Fresh stage, source/helper hashes, scheduler selection, one-shot ownership,
allocation/GPU evidence, exits, terminal event lifecycle, and outcome are
recorded below when each becomes durable.

### Terminal outcome

The corrected helper was installed through a verified same-directory atomic
rename at mode `0700`, owner `kwen1`, SHA-256
`9272af015d3db2f6a7dc244a4a3fb07e9854585ea14a1b9e379b2e497ec51e73`.
It created the ordinary independent stage
`/data/scratch-fast/kwen1/structured-rotation-secondcrew-gpu-smoke-r3/attempt-20260831T013733Z-f893845b9b91/stage/20260831T014957Z-6df3a6628026-code`
at prepared commit `6df3a662802686393034375c7cd7ccd568277611`. Source
and stage tracked manifests both hash to
`92ebfedc83d2234eae186ffa5fdcfac6a961f7fb937ec37bc85c07476edec397`;
source status, ordinary-untracked, and required allowed-ignored manifests were
empty. Stage-local `.git`/common-dir, no alternates, ancestry, tree, fsck,
expected metadata-only status, and temporary bundle absence all passed. The
independent post-stage audit is SHA-256
`73f9cfc26bf447f63c0fe62daa767c0f47fb332c0351dcfd2ea43e08fbcd6909`.

The live read-only suitability snapshot found all four Torralba GPU partitions
allowed for the account/QoS and both V100 nodes idle. The driver-only smoke
therefore selected `vision-torralba-v100`. After the atomic one-shot latch and
sole terminal event owner were durable, exactly one `salloc` and one
`srun --pty` were issued. Allocation `1638476` was granted on
`torralba-v100-1.csail.mit.edu` with the requested one node/task/GPU, two CPUs,
8 GiB, ten minutes, account `vision-torralba-urops-meng`, and QoS
`vision-torralba-interactive`.

The retained payload proves `SLURM_GPUS_ON_NODE=1`; `nvidia-smi -L` and the
one-line query both exited 0 and found exactly one Tesla V100-SXM2-32GB,
UUID `GPU-9abffda8-d89c-1c5d-321a-6989fa1c008a`, 32768 MiB, driver
`580.178.04`. However, `CUDA_VISIBLE_DEVICES` was empty, so the payload's
explicit assignment check computed `assigned_device_count=0` and returned 1.
That exit propagated through `srun`, `salloc`, SSH, and the local tmux owner.
This is a terminal failed smoke under the success contract despite the one-GPU
driver evidence; no retry is authorized and no scientific/performance claim is
made.

The sole event source
`when-structured-rotation-secondcrew-gpu-smoke-r3-terminal` fired as sequence
1 after 66 condition polls; `/usr/bin/true` exited 0. Its result was already
handled when reconciled and its remaining adapter records were then retired
idempotently. The durable marker records SSH exit 1 at
`2026-08-31T02:01:17Z`. Full hashes and exact argv are in
`/home/ubuntu/.treehouse/firstmate-557e63/5/firstmate/data/structured-rotation-secondcrew-gpu-smoke-r3/report.md`.
