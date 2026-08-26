# Structured-Hadamard research notebook

## 2026-08-25 — Phase A implementation preparation

### Provenance and ancestry

- Disposable source worktree:
  `/home/ubuntu/.treehouse/triton-ff92c5/3/triton`.
- Research branch: `fm/structured-hadamard-phase-a`.
- Exact clean base: `f893845b9b91599ebd3b7a9c7f28164f39c7ed94`.
- Origin: `https://github.com/kw7243/triton.git`.
- Intended upstream: `https://github.com/triton-lang/triton.git`.
- Triton license: MIT.
- Fork `main` at intake: the exact pinned base above. Official upstream `main`
  observed by the intake scout: `dda532b66d3376593069d726f907a444b912e874`;
  it was intentionally not imported or rebased into this provenance.
- Frozen semantic oracle: `https://github.com/kw7243/QuaRot`.
- Oracle upstream: `https://github.com/spcl/QuaRot.git`.
- Exact oracle commit: `5008669b08c1f11f9b64d52d16fddd47ca754c5a`.
- Oracle license: Apache-2.0.
- Chosen CSAIL active/output root: `/data/scratch-fast/kwen1`.
- Branch ancestry is pinned and must not be rebased: the exact base above is
  the ancestor of the delivery head on `fm/structured-hadamard-phase-a`. The
  hexadecimal implementation and delivery commits are appended below after
  validation; the final self-containing provenance commit is identified as
  `HEAD` because a Git commit cannot embed its own hash.

Isolation was verified before branching: both `pwd -P` and
`git rev-parse --show-toplevel` resolved to the disposable worktree above, and
`git rev-parse HEAD` returned the exact pinned base. The worktree began clean
and detached. No old `fm/phase-a-*` branch is in this ancestry.

### Scope and contracts

- First model contract: `meta-llama/Llama-2-7b-hf`, `d_model=4096`, online FFN
  `down_proj` input width `11008`, batch-1 decode first.
- Reference/spec coverage: `I`, normalized contiguous `H32`, normalized
  contiguous `H128`, and exact `Hfull=172x64`.
- Runnable Triton boundary in this phase: host-no-launch `I` and exact
  QuaRot-compatible `Hfull` only.
- Pinned `U_172` digest:
  `sha256:int8-row-major:378ef12c7cc31f3ea558e1c66b9552a83128f8f732b0bc094b864c8e120d84bd`.
- One fail-closed machine record contract:
  `rot-site-v1.phase-a.2` in
  `experiments/structured_hadamard/phase_a/schema.py`.
- CPU oracle: actual width 11008 with bounded token/weight rows, checking
  inverse and norm preservation, stored-weight fold orientation, local linear
  equivalence, and zero identity transform work.
- No zero-padding/truncation, random signs, permutations, learned stages, new
  INT4 GEMM, transformer port, selector, or fusion branch is in scope.

### Safety and evidence status

No experiment, benchmark, evaluation, model download, GPU job, CUDA command,
Slurm command, or scheduler operation has run. No W4A4 quality, latency,
affected-layer, end-to-end, PPL, or scientific result exists. The preflight
matrix is synthetic-input metadata only; every metric is null, every row is
marked `UNEXECUTED`, `scheduler_clearance=false`, and
`scientific_evidence=false`. Its `--execute` path refuses before any GPU import.

Future experimental commands must follow the research-reproducibility skill:
copy the complete repository first (including `.git`, local edits, configs, and
repo-local inputs), then execute only from that timestamped staged copy. The
source and durable staging/output root is `/data/scratch-fast/kwen1`. GPU work
also remains blocked on explicit scheduler clearance and a single named owner.

### Validation record

Dependency-free validation run from the source worktree on 2026-08-25:

```text
python3 -m compileall -q experiments/structured_hadamard/phase_a
  -> passed
python3 -m unittest discover -s experiments/structured_hadamard/phase_a/tests -v
  -> 14 tests passed in 1.048 seconds
python3 -m experiments.structured_hadamard.phase_a.oracle \
  --seed 0 --token-rows 2 --weight-rows 3
  -> all I/H32/H128/Hfull checks passed; maximum reported absolute error
     3.1086244689504383e-15; labeled CPU validation, not scientific evidence
python3 -m experiments.structured_hadamard.phase_a.preflight \
  --scheduler-clearance=false --format=jsonl \
  --result-jsonl /data/scratch-fast/kwen1/structured-hadamard/phase-a/results/phase-a.jsonl
  -> exact four rows validated: (transform-only, transform+quantize) x
     (I, Hfull), all scheduler_clearance=false
python3 -m experiments.structured_hadamard.phase_a.preflight \
  --scheduler-clearance=false --execute
  -> refused as required with exit status 2
```

The clean environment did not contain pytest, NumPy, Torch, Triton, ruff, or
yapf, and no dependency was installed. The suite therefore used only the
Python standard library. `make` was not run because the patch changes only
Python and documentation. No experimental command was run, so repository
staging was neither required nor performed in this task.

Pending final code-review gate and branch commit/push. Only dependency-free
static/unit/CPU checks are permitted in this task.

## 2026-08-26 — Captain-approved contract corrections

Resumed the existing `fm/structured-hadamard-phase-a` branch from verified
remote tip `172261a73c58ff0ce2818c52f9f63c4d1676941f`. The remote had not
drifted, the local branch matched it exactly, and
`f893845b9b91599ebd3b7a9c7f28164f39c7ed94` remained the exact merge base.
No branch recreation, reset, rebase, default-branch update, or ancestry change
was performed.

The corrected Phase A contract records `fusion="none"` for both timing
identities. `transform+quantize` means sequential transform-then-quantize
timing and makes no fused-kernel claim. Only `I` and `Hfull` measurement rows
are accepted; `H32` and `H128` remain CPU/reference specifications only.

Future experimental commands must use the repo-local
`experiments/structured_hadamard/phase_a/stage_repository.py` helper. It
materializes an ordinary, independent Git repository at the exact source
`HEAD`, copies and hashes every tracked and non-ignored untracked working-tree
input (including dirty tracked content), supports linked-worktree sources, and
fails before running a command unless those properties are re-verified. Its
preparation-time command is limited to a bounded Git assertion/no-op; no
benchmark or scientific workload is used to validate staging.

No experiment, benchmark, evaluation, GPU/CUDA/Slurm command, model use or
download, PR, merge, or no-mistakes pipeline has run as part of these
corrections. Final dependency-free test evidence and the delivery `HEAD` are
recorded after validation; as above, the self-containing delivery identity is
`HEAD` because a commit cannot include its own hash.

Final CPU/static validation from the source worktree on 2026-08-26:

```text
python3 -m compileall -q experiments/structured_hadamard/phase_a
  -> passed
python3 -m unittest discover -s experiments/structured_hadamard/phase_a/tests -v
  -> 26 tests passed in 1.895 seconds
python3 -m unittest experiments.structured_hadamard.phase_a.tests.test_stage_repository -v
  -> 3 tests passed in 0.639 seconds; ordinary-repository and linked-worktree
     fixtures each proved ordinary self-contained Git metadata, exact source
     HEAD, dirty tracked content, untracked content, and usability after the
     source metadata was removed; the optional command was only bounded
     `git cat-file -e HEAD^{commit}`
python3 -m experiments.structured_hadamard.phase_a.oracle \
  --seed 0 --token-rows 2 --weight-rows 3
  -> all I/H32/H128/Hfull CPU/reference checks passed at width 11008;
     maximum absolute error 3.1086244689504383e-15; explicitly not scientific
     evidence
python3 -m experiments.structured_hadamard.phase_a.preflight \
  --scheduler-clearance=false --format=jsonl \
  --result-jsonl /data/scratch-fast/kwen1/structured-hadamard/phase-a/results/phase-a.jsonl
  -> four unexecuted rows asserted: I/Hfull x transform-only/sequential
     transform+quantize; every row fusion=none, scheduler_clearance=false,
     scientific_evidence=false
python3 -m experiments.structured_hadamard.phase_a.preflight \
  --scheduler-clearance=false --execute
  -> refused with exit status 2 as required
git diff --check
  -> passed
```

`make` was not run because these corrections modify only Python and
documentation. No experimental command was run, so no actual research stage
or GPU/scheduler boundary was entered. The final ancestry remains exactly
`f893845b9b91599ebd3b7a9c7f28164f39c7ed94..HEAD` on the named branch.
