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
`HEAD`, copies and hashes every tracked and untracked working-tree input
(including Git-ignored repo-local inputs and dirty tracked content), supports
linked-worktree sources, and fails before running a command unless those
properties are re-verified. Its exact default exclusions are root-level
`staging`, `out`, `outputs`, `eval_outputs`, `slurm_outputs`, and `wandb`
trees, plus `.cache`, `__pycache__`, `.pytest_cache`, `.mypy_cache`,
`.ruff_cache`, `.venv`, and `venv` directories at any depth; tracked files
override every exclusion. Source `.git` metadata is cloned independently. Its
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

## 2026-08-26 — Ignored-input staging follow-up

Resumed the clean named branch with local and remote both at
`cd016b8e65ae81d3401e27492370ba93995313ac`. Read-only acceptance review found
that the staging manifest used Git's standard excludes and therefore omitted
ignored repo-local inputs. The helper now enumerates the working tree directly
and includes Git-ignored files in its hashed before/copy/after proof, subject
only to the exact generated/cache/virtualenv exclusions documented above.
The staging metadata contract is `phase-a-repository-stage-v2` and records the
active default exclusion lists. Existing independent-Git, exact-HEAD,
path-safety, object-connectivity, and source-stability checks are unchanged.

Follow-up CPU/static validation from the source worktree:

```text
python3 -m compileall -q experiments/structured_hadamard/phase_a
  -> passed
python3 -m unittest discover -s experiments/structured_hadamard/phase_a/tests -v
  -> 26 tests passed in 1.897 seconds
python3 -m unittest experiments.structured_hadamard.phase_a.tests.test_stage_repository -v
  -> 3 tests passed in 0.632 seconds; both ordinary and linked-worktree
     fixtures copied and verified an ignored repo-local input, excluded an
     ignored root outputs tree and nested __pycache__ tree, preserved exact
     HEAD and dirty tracked/ordinary untracked content (including a tracked
     file overriding the outputs exclusion), and remained usable after source
     metadata removal; the optional command remained the bounded
     `git cat-file -e HEAD^{commit}` assertion
python3 -m experiments.structured_hadamard.phase_a.oracle \
  --seed 0 --token-rows 2 --weight-rows 3
  -> all I/H32/H128/Hfull CPU/reference checks passed at width 11008;
     maximum absolute error 3.1086244689504383e-15; not scientific evidence
python3 -m experiments.structured_hadamard.phase_a.preflight \
  --scheduler-clearance=false --format=jsonl \
  --result-jsonl /data/scratch-fast/kwen1/structured-hadamard/phase-a/results/phase-a.jsonl
  -> four unexecuted I/Hfull rows passed the fusion=none, sequential-boundary,
     scheduler_clearance=false, and scientific_evidence=false assertions
python3 -m experiments.structured_hadamard.phase_a.preflight \
  --scheduler-clearance=false --execute
  -> refused with exit status 2 as required
git diff --check
  -> passed
```

`make` was not run because the follow-up changes only Python tests/helper code
and documentation. No experiment, benchmark, model, GPU/CUDA/Slurm command,
PR, merge, or no-mistakes pipeline ran. The delivery identity is the clean
named-branch `HEAD`, with the pinned base still its ancestor.

## 2026-08-26 — Staged scheduler-marker counterfactual preparation

Captain authorization is limited to task owner `rot-scheduler-marker-r1`, one
new complete repository stage, one marker-only `sbatch` invocation, and one
Firstmate-owned terminal event source. This is a scheduler-boundary
counterfactual, not the Phase A experiment. The submitted script must not run
Python, CUDA, `nvidia-smi`, model code, imports, benchmarks, or scientific
payloads, and there is no retry, cancellation, allocation, or manual polling
authority.

The accepted source and remote branch both began at
`c53cf746ade4c615aa3890c8be1b8b152df02038`. The commit containing this plan
and `experiments/structured_hadamard/phase_a/run_scheduler_marker.sh` will be
the exact pinned source for staging. The marker token is
`e27feb7a0c3cd46a8a4bc35c70c87591`. After its shebang and Slurm comments, the
script's first executable action is the single `printf` that writes the token,
Slurm job id, and exported absolute stage path to combined durable Slurm
stdout/stderr; its next executable action is `exit 0`.

The recovered failed-staged contract is held fixed: job name
`phase-a-hurwitz`; account `vision-torralba-urops-meng`; QoS
`vision-torralba-interactive`; partition `vision-torralba-rtx3090`; one node,
one task, one `gpu:1`, four CPUs, `16G`, and `01:00:00`; submission working
directory equal to the new stage; explicit
`ALL,RESEARCH_REPRO_STAGED_DIR=<stage>,SOURCE_REPO=<source>` exports; and
combined output/error pattern
`/data/scratch-fast/kwen1/compute-native-vq/triton/results/2026-08-20-hurwitz-decode-baseline/slurm-%j.out`.
As in failed staged retry 3, account/QoS/partition are repeated on the `sbatch`
CLI while the job shape remains in the script. The only authorized launch
changes are the new compliant full stage and the absolute staged marker-script
argument.

Before submission, the repo-local `stage_repository.py` helper must create a
new stage under `/data/scratch-fast/kwen1`, and read-only preflight must prove
the exact source/staged commit, ordinary independent staged `.git`, object
connectivity, complete `phase-a-repository-stage-v2` input manifest, source
stability, marker bytes/mode/action order, scratch/output ownership and modes,
non-colliding `%j` output pattern, exact `sbatch` argv, and exclusive task
ownership. A durable latch must reserve the sole attempt and the deterministic
Firstmate source-id mapping `slurm-<canonical numeric job id>` before the one
submission. Any staging/preflight failure stops before `sbatch`; an ambiguous
submission consumes the attempt and permits neither a retry nor a scheduler
query.

On a unique numeric id, durable remote provenance and this source notebook
will record the stage, commits, manifest evidence, exact argv, output path,
marker, job id, and concrete `slurm-<job id>` handoff. Firstmate alone will arm
that terminal source. This task owner will not query or wait on the job and
will append terminal evidence only after a later Firstmate steer.

### Sole submission and terminal-owner handoff

The exact pinned source and staged commit are both
`c93a31f7a9343f8ceeaf90d3955bae1104c7b122`, a normal descendant of the
accepted base. The clean remote source is
`/data/scratch-fast/kwen1/structured-hadamard/rot-scheduler-marker-r1/source-c93a31f7a934`.
The newly created stage is
`/data/scratch-fast/kwen1/structured-hadamard/rot-scheduler-marker-r1/staging/20260826T204344Z-c93a31f7a934-code`.
Both are owned by `kwen1` (UID 28131). The stage has an ordinary self-contained
`.git` directory, no object alternate, exact `HEAD`, verified connectivity,
and remains usable independently of the source metadata.

The stage metadata schema is `phase-a-repository-stage-v2`. Its complete
working-tree manifest contains 1,753 entries with digest
`bfa8f8e2a03876347c755460097b80c08ab8d1fe2db6dea3912993b1ab377700`;
`REPRODUCIBILITY_METADATA.json` has SHA-256
`b634b6c55b256a2fb3de038dad778a57ad40c6cba14d5de6fb7f4b2a5f8905b6`.
The source was stable and clean, so the required dirty-tracked, ordinary
untracked, and allowed ignored included-input counts were each zero. The full
path/identity manifest and preflight summary are preserved at mode 0600 under
`/data/scratch-fast/kwen1/structured-hadamard/rot-scheduler-marker-r1/run-state/submit-once.latch/`.
The helper's earlier ordinary- and linked-worktree tests separately exercised
nonzero dirty, untracked, and ignored inputs before this stage was trusted.

The absolute submitted marker path is
`/data/scratch-fast/kwen1/structured-hadamard/rot-scheduler-marker-r1/staging/20260826T204344Z-c93a31f7a934-code/experiments/structured_hadamard/phase_a/run_scheduler_marker.sh`.
Preflight proved 696 bytes, executable mode 0775, SHA-256
`78787a8883f6fd257a7e3dc48aed6eac6717fb109eccbc45652a97a77583bd28`,
and exactly two executable actions: the first is the single `printf` of marker
`e27feb7a0c3cd46a8a4bc35c70c87591`, `$SLURM_JOB_ID`, and
`$RESEARCH_REPRO_STAGED_DIR`; the second is `exit 0`. The script contains no
scientific or diagnostic payload.

The output parent was owned by `kwen1`, mode 0755, writable/searchable, and had
zero pre-existing `slurm-*.out` files. Combined output/error uses the preserved
template
`/data/scratch-fast/kwen1/compute-native-vq/triton/results/2026-08-20-hurwitz-decode-baseline/slurm-%j.out`.
The exact submission working directory was the stage above. The exact sole
`sbatch` argv was:

```text
/usr/bin/sbatch --parsable --account=vision-torralba-urops-meng --qos=vision-torralba-interactive --partition=vision-torralba-rtx3090 --export=ALL,RESEARCH_REPRO_STAGED_DIR=/data/scratch-fast/kwen1/structured-hadamard/rot-scheduler-marker-r1/staging/20260826T204344Z-c93a31f7a934-code,SOURCE_REPO=/data/scratch-fast/kwen1/structured-hadamard/rot-scheduler-marker-r1/source-c93a31f7a934 /data/scratch-fast/kwen1/structured-hadamard/rot-scheduler-marker-r1/staging/20260826T204344Z-c93a31f7a934-code/experiments/structured_hadamard/phase_a/run_scheduler_marker.sh
```

The durable latch reserved attempt limit 1, submission owner
`rot-scheduler-marker-r1`, and the deterministic terminal-source rule before
the invocation. The one captured invocation returned status 0 and exactly one
canonical numeric stdout line: Slurm job `1589782`. Captured stdout, stderr,
and exit status remain in the latch; they were not used to authorize any retry.
The exact realized output path is
`/data/scratch-fast/kwen1/compute-native-vq/triton/results/2026-08-20-hurwitz-decode-baseline/slurm-1589782.out`.

Terminal-state ownership is handed exclusively to Firstmate as source id
`slurm-1589782`. The task status is paused on that registered-owner handoff,
not complete. This owner did not poll, wait on, inspect, cancel, requeue, or
otherwise query or mutate the accepted job after submission, and will never
submit another job for this task. A later Firstmate steer is required to append
the event-captured terminal outcome here.

### Event-owned terminal result and bounded conclusion

Firstmate's sole terminal owner fired once, was handled once, and is retired.
The actual existing deterministic adapter source was `when-slurm-1589782`
(one condition poll; terminal action exit 0). This supersedes the pre-submit
generic `slurm-1589782` source-id reservation with the exact source Firstmate
registered; no competing source existed.

The captured terminal allocation record for job `1589782` is `CANCELLED by
UID 0` with `ExitCode=0:0`. It ran on `torralba-3090-3`; the supplied scheduler
timestamps are start `2026-08-26T16:52:20`, end
`2026-08-26T16:52:23`, and elapsed `00:00:03`. The exact allocation was
`billing=1044,cpu=4,gres/gpu=1,mem=16G,node=1`. Its WorkDir was the exact stage
recorded above, and StdOut/StdErr retained the established combined
`slurm-%j.out` template.

The concrete output
`/data/scratch-fast/kwen1/compute-native-vq/triton/results/2026-08-20-hurwitz-decode-baseline/slurm-1589782.out`
is missing. Therefore the marker script's first executable action produced no
durable marker, and there is no evidence that user script execution began.

This is scheduler-boundary evidence, not scientific evidence. Because the
submitted script contained only the first-action marker write followed by
`exit 0`, this counterfactual excludes Python, Triton, CUDA, model code,
imports, benchmarks, and the Phase A scientific payload as necessary causes
of this cancellation. A newly compliant full stage and absolute staged script
did not avoid the same pre-output UID-0 cancellation class. The evidence does
not distinguish working-directory entry, output opening, script exec,
prolog/cgroup/GRES setup, daemon/controller behavior, or another privileged
trigger; it does not identify an actor or root cause, prove the stage itself
causal, provide scheduler clearance, or produce any Phase A measurement.

The durable terminal record is
`/data/scratch-fast/kwen1/structured-hadamard/rot-scheduler-marker-r1/run-state/submit-once.latch/terminal-event.txt`.
It is owned by `kwen1`, mode 0600, with SHA-256
`ec436b343676210a1c4dee0e1943b9418d3c40828da8704e1588df68ee8f9ece`.
No scheduler query or mutation, second submission, retry, PR, or merge followed
the event-owned result.

## 2026-08-31 — Submission-free Phase A execution driver preparation

The accepted remote source `origin/fm/structured-hadamard-phase-a` was fetched
and its tip was verified unchanged at
`d57acb60db2a4507bbff984fb3c9771e8a6ada3d` before branching. The isolated
worktree was clean and detached. The sole new local research branch is
`fm/structured-hadamard-phase-a-exec-r1`, created directly from that exact
commit without reset, rebase, or ancestry substitution. This intake commit is
the exact base of the delivery `HEAD`; the final hexadecimal `HEAD` is reported
after commit because a commit cannot embed its own identity.

The new `execute.py` boundary is runnable only from a clean, ordinary,
self-contained stage descended from the accepted intake. It verifies the
executed `HEAD`, Git object connectivity/independence, the stage metadata,
the complete working-tree manifest, and exact committed contents both before
and after the CPU correctness oracle. A mode-0600 owner clearance outside the
stage must bind the stage/source/driver/transform commit, manifest digest,
explicit `cuda:N` device, immutable model and calibration revisions, resolved
W4A4 settings, fixed timing config, and a new canonical durable output
directory. The driver owns no resource acquisition, remote access, job
control, retry, or lifecycle behavior. The existing submission-free
`preflight.py --execute` refusal remains active.

`stage_repository.py` now emits `phase-a-repository-stage-v3` metadata plus an
explicit `phase-a-working-tree-manifest-v1`
`REPRODUCIBILITY_MANIFEST.json`; the metadata binds both its file digest and
canonical entry digest. Execution additionally requires that this manifest
contain exactly committed tracked paths, thereby refusing dirty or untracked
executed inputs even though the general staging helper continues to preserve
such inputs for other reproducibility uses.

The only added quantization callback is a one-launch deterministic dynamic
per-row symmetric signed-A4 fake-quantize/dequantize operation with
round-to-nearest-even and range `[-7,7]`. It supplies the sequential
`transform+quantize` timing boundary and does not implement weight conversion,
nibble packing, an INT4 GEMM, a model/checkpoint path, or fusion. Measurement
assembly remains exactly `I`/`Hfull` crossed with `transform-only`/sequential
`transform+quantize`; `I` remains a host alias with zero transform launch/copy,
`Hfull` remains the exact two-launch `11008=172x64` transform, and every row
records `fusion="none"`. The output writer validates four complete finite
`rot-site-v1.phase-a.2` rows and four raw-sample rows before atomic durable
writes. It records observed runtime identities, commit/config/clearance
digests, launch counts, correctness/error/outlier metrics, and raw samples;
all rows are visibly synthetic, non-model, non-PPL, and
`scientific_evidence=false`.

Final CPU/static validation from the source worktree:

```text
python3 -m compileall -q experiments/structured_hadamard/phase_a
  -> passed
python3 -m unittest discover -s experiments/structured_hadamard/phase_a/tests -v
  -> 36 tests passed in 4.130 seconds
python3 -m unittest experiments.structured_hadamard.phase_a.tests.test_execute -v
  -> 7 focused execution/provenance/refusal/assembly tests passed in 2.245 seconds
python3 -m unittest experiments.structured_hadamard.phase_a.tests.test_stage_repository -v
  -> 3 temporary ordinary/linked-worktree staging tests passed in 0.896 seconds
python3 -m experiments.structured_hadamard.phase_a.oracle \
  --seed 0 --token-rows 2 --weight-rows 3
  -> all I/H32/H128/Hfull CPU/reference checks passed at width 11008;
     maximum absolute error 3.1086244689504383e-15; not scientific evidence
python3 -m experiments.structured_hadamard.phase_a.preflight \
  --scheduler-clearance=false --format=jsonl \
  --result-jsonl /data/scratch-fast/kwen1/structured-hadamard/phase-a/results/phase-a.jsonl
  -> exactly four unexecuted I/Hfull rows validated
python3 -m experiments.structured_hadamard.phase_a.preflight \
  --scheduler-clearance=false --execute
  -> refused with exit status 2
python3 -m experiments.structured_hadamard.phase_a.execute \
  --scheduler-clearance-file /tmp/phase-a-clearance-does-not-exist
  -> refused outside an independent stage with exit status 2, before reading
     clearance or importing Torch/Triton
git diff --check
  -> passed
```

Pytest was unavailable and no dependency was installed. `make` was not run
because the changes are Python, tests, and documentation only. The staging
tests created and removed only bounded temporary fixture repositories; no
durable experimental stage was created. No GPU was allocated, no CUDA command
or kernel ran, no Slurm operation occurred, no model/checkpoint was downloaded
or used, and no benchmark, PPL evaluation, scientific experiment, push, PR,
merge, or no-mistakes pipeline ran.

## 2026-08-31 — Phase A compatible-GPU kernel microprofile preflight

Task `rot-phasea-kernel-profile-r1` began in the isolated task worktree on the
new local-only branch `fm/structured-hadamard-phase-a-kernel-profile-r1`,
created directly at exact parent
`21ea761c61a5e3062cea28cabda43ac04bd5278b`. That parent has exact parent
`d57acb60db2a4507bbff984fb3c9771e8a6ada3d`, the accepted Phase A commit, and
`git merge-base --is-ancestor` returned 0. The accepted execution contract
measures only `I` and `Hfull`; `H32`/`H128` remain CPU/reference-only;
`transform+quantize` is sequential transform-then-quantize with
`fusion="none"`; and every output remains synthetic, non-model, non-PPL,
non-end-to-end evidence with `scientific_evidence=false`.

The required branch initially exposed one bounded preflight defect: the exact
parent's `SOURCE_BRANCHES` allowlist did not include this task's required branch,
so otherwise-valid CPU preflight exited 1 before producing its matrix. The
repair adds only the required branch and an executable identity-discovery test;
it does not modify the CUDA execution driver or scientific boundary. A separate
environment divergence showed that `/data` is not mounted on this local task
host. This is not a Phase A failure: durable scratch work must use the proven
persistent SSH route to CSAIL, and no scheduler mutation occurred.

Bounded local CPU/static validation after the repair:

```text
python3 -m compileall -q experiments/structured_hadamard/phase_a
  -> passed
python3 -m unittest discover -s experiments/structured_hadamard/phase_a/tests -v
  -> 37 tests passed in 4.891 seconds
python3 -m unittest experiments.structured_hadamard.phase_a.tests.test_execute -v
  -> 7 tests passed in 2.300 seconds
python3 -m unittest experiments.structured_hadamard.phase_a.tests.test_stage_repository -v
  -> 3 tests passed in 1.117 seconds
python3 -m unittest experiments.structured_hadamard.phase_a.tests.test_schema -v
  -> 13 tests passed in 0.024 seconds
python3 -m experiments.structured_hadamard.phase_a.oracle --seed 0 --token-rows 2 --weight-rows 3
  -> all I/H32/H128/Hfull CPU/reference checks passed; maximum absolute error
     3.1086244689504383e-15; not scientific evidence
python3 -m experiments.structured_hadamard.phase_a.preflight --scheduler-clearance=false --format=jsonl \
  --result-jsonl /data/scratch-fast/kwen1/structured-hadamard/rot-phasea-kernel-profile-r1/outputs/phase-a.jsonl
  -> exactly four I/Hfull unexecuted rows validated
python3 -m experiments.structured_hadamard.phase_a.preflight --scheduler-clearance=false --execute
  -> refused with exit status 2
python3 -m experiments.structured_hadamard.phase_a.execute \
  --scheduler-clearance-file /tmp/rot-phasea-kernel-profile-r1-clearance-missing
  -> refused with exit status 2 before Torch/Triton import
git diff --check
  -> passed
```

`make` was not run because this preflight changes only Python/tests/notebook.
Torch and Triton were not imported and no GPU code ran on a login node. Before
staging, these changes will be committed so the source branch is clean. The
fresh remote source/stage identity and exact commit will be recorded after the
commit exists.

This task preserves the r3 conclusion exactly: allocation `1638476` reached
`torralba-v100-1` and exposed one V100 through successful `nvidia-smi`, while
its strict payload failed because `CUDA_VISIBLE_DEVICES` was empty and
assigned-device count was zero. This run is scientifically distinct: it will
not create or retry a generic visibility smoke and will accept only the
SM86/RTX 3090 target.
