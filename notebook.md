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

### Pre-allocation terminal blocker

Pinned source commit `a8e10f61ba60cd26cb360f133e39527c635c5ee4` is a
descendant of the base above. Its clean working tree had zero dirty, ordinary
untracked, or ignored input files. The complete branch bundle and tracked
manifest were written under
`/data/scratch-fast/kwen1/structured-rotation-secondcrew-gpu-smoke-r2/attempt-20260830T231328Z-a8e10f61ba60/preflight/`.
Their SHA-256 values are respectively
`ef3e69116a7a39f6780abdcf1cd59655190d2fe06ef56d1d99c117c9f8fd6e94`
and `01882f86059c36d9b9867de636ab0b4600ca5838480d945156f382126b05d541`;
the empty source status and ignored-input manifests each have the canonical
empty-file digest
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
The payload digest is
`c25a8d974c97591690b37133d1ea7dd0751f9017a938e64499a211e68a55d0f2`.
The only excluded source material was linked-worktree `.git` administrative
metadata, which the complete bundle was intended to replace with an ordinary
independent clone; no working-tree input was excluded.

The exact task-owned tmux SSH route was
`structured-rotation-secondcrew-gpu-smoke-r2-ssh`. Before its release, the
one-shot local latch and sole terminal owner
`when-structured-rotation-secondcrew-gpu-smoke-r2-terminal` were armed. All
remaining remote writes used that persistent route. Materialization then
failed closed twice on the same condition: the staged helper called
`git bundle verify` outside a repository, whose durable output is
`preflight/bundle-verify.txt` (`error: need a repository to verify a bundle`).
The helper had sealed itself mode `0500`; the corrected upload could not
replace those stale bytes, so the stdin execution repeated the same verify
failure. The task's two-strike rule prohibited another correction or attempt.

The durable blocked latch is
`run-state/allocation-once.latch/` below the attempt root. Its `blocked.env`
has SHA-256
`47c8c8e94fd008722c7bdf3919bf6fab9cfa620ce1f831d7ed776887c6e3ea11`;
`preflight/allocation-latch-files.sha256` records all exact argv and latch
hashes. It records `salloc_attempts=0`, `srun_attempts=0`, and
`allocation_granted=false`. Consequently there is no allocation/job id,
compute hostname, CUDA visibility, `nvidia-smi` output, GPU identity, or
payload/SSH success result, and no scientific or performance claim.

The exact tmux-owned SSH route exited naturally at `2026-08-30T23:22:13Z`
with propagated SSH exit code `2`. Its sole event owner fired as sequence `1`
after 140 condition polls; the harmless `/usr/bin/true` action exited `0`.
Firstmate recorded the handled acknowledgement and retired the source, and no
registration remains. The event result has SHA-256
`990d95eacb152ef4221863414469fa8cf2533290b4d373e0c8823d542aa412f2`.
The exact terminal outcome is `BLOCKED_PRE_ALLOCATION`: allocation/job id
`none`, selected partition `vision-torralba-v100`, zero scheduler attempts,
and a handled/retired terminal transport event.
