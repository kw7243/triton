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
