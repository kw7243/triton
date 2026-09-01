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

### Immutable stage, clearance, and pre-allocation terminal result

The clean pre-run/source/stage commit is
`ddb738a07a098df48e62e84f397f023721c2f55a`, tree
`c953b2829cdc2d6a703cc8292392aa217c6a80db`, with exact direct parent
`21ea761c61a5e3062cea28cabda43ac04bd5278b`; accepted Phase A commit
`d57acb60db2a4507bbff984fb3c9771e8a6ada3d` remains its ancestor. The full
source bundle SHA-256 is
`7e1d885ac7a80500e84a78af8b20cf865131ef71758f64498af0da58b077aa13`.
Bundle verification ran successfully both locally and remotely from inside
real disposable Git repositories. The source clone is clean with zero dirty,
ordinary-untracked, or ignored entries.

The one fresh complete stage is:

```text
/data/scratch-fast/kwen1/structured-hadamard/rot-phasea-kernel-profile-r1/attempt-20260831T162141Z-ddb738a07a09/stage/20260831T162141Z-ddb738a07a09-code
```

It has one ordinary self-contained worktree and stage-local `.git`, no
alternates/promisor/shallow dependency, successful connectivity verification,
exact source/stage commit and tree, and exactly the two declared untracked
control files. Its 1,756-entry canonical manifest digest is
`0f1243905034afbc6028324ca90bf9aaa50dd2f98f68131ff9cf72bca44a4169`;
the manifest file SHA-256 is
`518d8348eafdbc9f73c02ff799f7bb75b1ec936f22142573151739aec024aabc`
and metadata SHA-256 is
`e423c819eadf28b72d089823cb9ee628140e85e9954d997acc4f56cb9c290567`.
The staged helper and driver SHA-256 values are respectively
`212bc2ddd82ee2f098314140e9ec62032b3dbf350f421d61a658c812a08f7a08`
and `a2f968b1f6efa1bc1838520802a8da9fc577ec14df5326b792c36048c6bff659`;
each matches its exact Git blob. The final independent static stage audit is
mode 0600, SHA-256
`41154764fcc6d40f8b185506ee5b76dd943fee16054b493bbc2164428864e3aa`,
and proves that neither Torch nor Triton was imported. Two earlier audit-tool
failures (nested-shell quote loss and an explicit external-script import-path
miss) remain preserved rather than overwritten; neither changed stage bytes.

The driver-validated owner-only clearance is
`.../attempt-20260831T162141Z-ddb738a07a09/run-state/clearance.json`, owned by
UID 28131 at mode 0600 with SHA-256
`d6bc3e622b602949008dd1fbe64429443eeb078cea1bc24bf88c1cdbd68c5b09`.
It binds the exact stage/commit/manifest, `cuda:0`, nonexistent output
`.../outputs/phase-a-ddb738a07a09-sm86`, synthetic `[1,11008]` seed-0
workload, 25 ms warmup, 200 ms repetition, five outer trials, resolved W4A4
metadata, model repository pin
`01c7f73d771dfac7d292323805ebc428287df4f9`, and Salesforce/wikitext pin
`b08601e04326c79dfdd32d625aee71d232d685c3`. No model or dataset bytes were
downloaded. Its static audit SHA-256 is
`3834547c1caaa8953f38decfda65746ea036b2102f57bbefe7057d0eae84d80a`.

The read-only scheduler snapshot at `2026-08-31T16:33:15Z` confirmed account
`vision-torralba-urops-meng` and QoS `vision-torralba-interactive`, then
reported every compatible node unavailable:

```text
torralba-3090-1  Gres=gpu:rtx_3090:7  GresUsed=gpu:rtx_3090:7(IDX:0-6)
torralba-3090-2  Gres=gpu:rtx_3090:7  GresUsed=gpu:rtx_3090:7(IDX:0-6)
torralba-3090-3  Gres=gpu:rtx_3090:7  GresUsed=gpu:rtx_3090:7(IDX:0-6)
```

The snapshot SHA-256 is
`8dbf3d3f32c8d10e56baef0decb46d1a0892bbd4860671a2f1b26b1949ceb4f4`.
Because the accepted target requires SM86/RTX 3090, V100 was not substituted
and H100/H200 were not preferred. The contract therefore stopped before
mutation: `salloc_attempts=0`, `srun_attempts=0`, no persistent allocation
owner or process-event source was armed, and no GPU/CUDA/Triton kernel,
correctness invocation, timed row, output row, or important-kernel overhead
gate ran. This is a faithful pre-allocation `blocked` result, not synthetic
kernel evidence.

All retained attempt evidence is rooted at:

```text
/data/scratch-fast/kwen1/structured-hadamard/rot-phasea-kernel-profile-r1/attempt-20260831T162141Z-ddb738a07a09
```

The 62-entry evidence checksum manifest is mode 0600 with SHA-256
`fc3ed7ad445f56d72cb3862e1edff6168a220cbbc2227b9498ef32f7dfe0cc40`.
No scheduler job, protected job `1579631`, active CNVQ owner/job, push, PR,
merge, force, retry, cancellation, requeue, model/PPL/end-to-end path, Phase
B/C path, or H32/H128 GPU path was touched.

## 2026-09-01 — Compatible-GPU relaunch

The captain revised `rot-phasea-kernel-profile-r1` to accept any currently
available suitable Torralba GPU, dynamically choosing the smallest adequate
and least scarce compatible accelerator. The required branch remains
`fm/structured-hadamard-phase-a-kernel-profile-r1`; its exact intake parent is
`21ea761c61a5e3062cea28cabda43ac04bd5278b`, and
`d57acb60db2a4507bbff984fb3c9771e8a6ada3d` is an ancestor of that parent.
The prior RTX-3090-only availability source fired, was handled, and retired;
both allocation ledgers remain `salloc_attempts=0` and `srun_attempts=0`.

The prior immutable stage and clearance remain preserved evidence but are not
eligible for this relaunch: their output identity and clearance were bound to
SM86 before the actual accelerator could be selected. The relaunch therefore
requires a new clean commit, a fresh full independent repository stage, a new
owner-only clearance bound to the selected GPU request and output, and a new
single persistent owner/process-event source. The accepted driver itself is
architecture-generic at its boundary: it selects an explicit `cuda:N`, records
the observed device compute capability, and contains no RTX-3090/SM86
assertion. Actual compatibility still must be proved inside the sole
allocation before the one driver invocation.

The scientific boundary is unchanged: measure exactly `I` and `Hfull` for
`transform-only` and sequential `transform+quantize`, with `fusion="none"`.
`H32` and `H128` remain CPU/reference-only. The result, if obtained, is
synthetic kernel evidence with `scientific_evidence=false`, not model, PPL, or
end-to-end evidence. The important-kernel gate applies only to same-device
`Hfull` versus `I` transform-plus-quantize overhead. A non-RTX-3090 result
cannot establish RTX-3090-specific latency or generalize the cost gate across
architectures.

Bounded CPU/static validation for the revised pre-run contract completed from
the source worktree before staging:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m compileall -q experiments/structured_hadamard/phase_a
  -> passed
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s experiments/structured_hadamard/phase_a/tests -v
  -> 37 tests passed in 3.224 seconds
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest experiments.structured_hadamard.phase_a.tests.test_execute -v
  -> 7 tests passed in 1.444 seconds
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest experiments.structured_hadamard.phase_a.tests.test_stage_repository -v
  -> 3 tests passed in 0.688 seconds
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest experiments.structured_hadamard.phase_a.tests.test_schema -v
  -> 13 tests passed in 0.032 seconds
PYTHONDONTWRITEBYTECODE=1 python3 -m experiments.structured_hadamard.phase_a.oracle \
  --seed 0 --token-rows 2 --weight-rows 3
  -> all I/H32/H128/Hfull CPU/reference checks passed; maximum absolute error
     3.1086244689504383e-15; not scientific evidence
PYTHONDONTWRITEBYTECODE=1 python3 -m experiments.structured_hadamard.phase_a.preflight \
  --scheduler-clearance=false --format=jsonl \
  --result-jsonl /data/scratch-fast/kwen1/structured-hadamard/rot-phasea-kernel-profile-r1/outputs/phase-a.jsonl
  -> exactly four I/Hfull rows; no CUDA import or GPU work
PYTHONDONTWRITEBYTECODE=1 python3 -m experiments.structured_hadamard.phase_a.preflight \
  --scheduler-clearance=false --execute
  -> refused with exit status 2
PYTHONDONTWRITEBYTECODE=1 python3 -m experiments.structured_hadamard.phase_a.execute \
  --scheduler-clearance-file /tmp/rot-phasea-kernel-profile-r1-clearance-missing
  -> refused with exit status 2 before Torch/Triton import
git diff --check
  -> passed
```

`make` was not run because the relaunch changes only this notebook. No CUDA,
GPU, model, dataset, or scheduler operation was performed by these gates.

### Fresh stage, sole hardware attempt, and terminal failure

The clean pre-run commit is
`11c6d87772de27d97a7d5b1f1f9577d70b4b41ca`, tree
`385c334f319c7069c1d81c2ce8db85e03e13731f`, descended through the preserved
task history from exact required parent
`21ea761c61a5e3062cea28cabda43ac04bd5278b`; the accepted Phase A commit remains
an ancestor. The full bundle SHA-256 is
`95ec5964c3776fc6e44debbcb251504623bf6b5c1ea3bf138deffb46e5d53686`.
It verified locally and remotely from real disposable Git repositories.

The fresh full independent stage is:

```text
/data/scratch-fast/kwen1/structured-hadamard/rot-phasea-kernel-profile-r1/attempt-20260901T164930Z-11c6d87772de/stage/20260901T164930Z-11c6d87772de-code
```

Its ordinary stage-local `.git`, connectivity, exact commit/tree, source
stability, and manifest all passed the final independent audit. The manifest
has 1,756 committed entries with canonical digest
`ec8e4c6f926bef98441b13678bcd6024b4ded27287f8ba96865889215c44045b`;
the manifest file SHA-256 is
`365307bac05af7c7af7707a275118efc0b653910270a2d3e895d7e1f38d80d85`
and metadata SHA-256 is
`83e1d7e600680dcd685a1c609e59f92a03374aa91b4c2f122b630e2f914a62cc`.
Only the two declared stage control files are untracked. The source clone was
clean with zero ordinary-untracked inputs; six helper-generated pycache files
were within the declared excluded cache policy. Two earlier external audit
failures are preserved: one wrapper used the AFS cwd and failed before helper
import/execution, and two audit assertions incorrectly resolved relative Git
paths and required zero excluded pycache files. The successful helper and
stage bytes were not rerun or mutated by those checker corrections.

The new clearance is mode 0600, owned by UID 28131, and has SHA-256
`48fb5cb360d57de604ea725318a43f878f2a797013a97b34b39f6af33c72318c`.
It passed the staged driver's static authorization path without Torch/Triton
import and bound the exact commit/stage/manifest, `cuda:0`, nonexistent output,
fixed `[1,11008]` seed-0 synthetic input, 25 ms warmup, 200 ms repetition,
five outer trials, immutable model/dataset metadata pins, and resolved W4A4
metadata. No model or dataset bytes were downloaded.

The read-only scheduler snapshot at `2026-09-01T16:57:49Z`, SHA-256
`458cb15065dc91b222704df7f0c3b7a506acaa2efc49127e3ac766e4f2dee6cd`,
found six free 24 GiB RTX 3090 devices on `torralba-3090-2`.
`vision-torralba-rtx3090` was selected without a node pin because it was the
smallest-memory suitable Torralba partition allowed by both the required
account and interactive QoS. V100/H100/H200 had more memory; shared smaller
devices did not allow that QoS.

One local-tmux-owned SSH route made exactly one `salloc` and one `srun --pty`
attempt. Allocation `1659613` was granted on `torralba-3090-2` with the exact
requested one node/task/GPU, two CPUs, 8 GiB, and ten minutes.
`CUDA_VISIBLE_DEVICES` was non-empty (`0`), and `nvidia-smi` recorded one
NVIDIA GeForce RTX 3090 (Ampere/SM86), UUID
`GPU-82e6108f-1eeb-47f3-d820-ad67a7a6bb15`, 24,576 MiB VRAM, and driver
`580.178.04`. The runtime was Torch `2.8.0+cu128`, CUDA `12.8`, and Triton
`3.4.0`. Control flow reached the correctness call only after exactly-one-device
Torch validation and a synchronized real CUDA tensor operation. The final
hardware JSON was emitted only on complete preflight success, so the observed
compute-capability tuple was not serialized before the later failure.

The one untimed Triton correctness invocation failed on the first A4
quantizer compilation, before `Hfull` correctness and before the accepted
driver invocation:

```text
AttributeError: module 'triton.language.extra' has no attribute 'libdevice'
```

The exact staged callback calls `tl.extra.libdevice.rint`; installed Triton
3.4.0 exposes the CUDA implementation containing `rint` under
`tl.extra.cuda.libdevice`. This is a driver/runtime API compatibility failure,
not the r3 visibility failure. The authorized attempt is consumed with
`salloc_attempts=1` and `srun_attempts=1`. The output directory is absent;
there are no raw samples or summaries for any of the four rows, and the
same-device transform-plus-quantize `Hfull` versus `I` overhead and `>=5%`
important-kernel gate are not evaluated. The identity transform-only no-op is
not used as a denominator. No RTX-3090 latency/overhead or cross-architecture
hardware-gate conclusion exists.

The single terminal marker fired once, was handled once, and was retired at
`2026-09-01T17:04:58Z`; the persistent owner then closed. No retry, Slurm
poll/query, cancellation, requeue, second owner, `sbatch`, protected job
inspection, CNVQ work, model/PPL/end-to-end path, Phase B/C, or H32/H128 GPU
path followed. The terminal event SHA-256 is
`fb209cb7b997a71da58e05a1d0ea8a4e3fe405cd1b9fd3c4e8473200dad6c131`;
GPU log SHA-256 is
`1c76caa84094792e5622999ed5629d84e914700f6e3f3d74b4cfa86aed3e4284`;
tmux log SHA-256 is
`0aa39af6506f1873d52a208a62fcb7a2a67dba674cccd6e2bf3dad0c8905cb86`.
The 55-entry evidence manifest is mode 0600 with SHA-256
`db23c4288c03b5420d04800f6aadcd01e620e66d1df01bc1caf7a41b3aed1594`.
The concise report is `data/rot-phasea-kernel-profile-r1/report.md`, SHA-256
`b9fc99a82c88d88c03850d99c27437650db35671fc6b6ecfbfdeba85e030a427`;
its final reporting commit is `HEAD` because a commit cannot embed its own
hash.

This task preserves the r3 conclusion exactly: allocation `1638476` reached
`torralba-v100-1` and exposed one V100 through successful `nvidia-smi`, while
its strict payload failed because `CUDA_VISIBLE_DEVICES` was empty and
assigned-device count was zero.

## 2026-09-01 — Triton 3.4 libdevice compatibility diagnosis

Task `rot-phasea-libdevice-fix-gpu-r2` began on local-only branch
`fm/structured-hadamard-phase-a-libdevice-fix-gpu-r2` at exact failed-run
evidence commit `095cafd6637238c53223f44d48b36b6f5211186b`, tree
`c47fedfce780e9f328d97e06e60373883e29ea60`. Both required
`git merge-base --is-ancestor` checks passed for accepted Phase A commit
`d57acb60db2a4507bbff984fb3c9771e8a6ada3d` and execution-driver parent
`21ea761c61a5e3062cea28cabda43ac04bd5278b`.

Gate 1 used only read-only SSH to `slurm-login.csail.mit.edu` and CPU-side
Python/import/compiler probes. It made no Slurm query or mutation, imported no
Torch or CUDA runtime module, and executed no GPU code. The exact environment
Python is
`/data/scratch-fast/kwen1/micromamba/root/envs/causal_forcing/bin/python`
(Python 3.10.20). `importlib.metadata.distribution("triton")` and the imported
package both report Triton `3.4.0`; the distribution directory is
`.../site-packages/triton-3.4.0.dist-info`, installed by `pip` from a CPython
3.10 manylinux 2.27/2.28 wheel. `direct_url.json` is absent. Relevant exact
installed bytes are:

```text
METADATA sha256=8496b57becd6f0fe732bf81570330383f25e2bc572a54bdec45671a23e1db189
INSTALLER sha256=ceebae7b8927a3227e5303cf5e0f1f7b34bb542ad7250ac03fbcde36ec2f1508
WHEEL sha256=39bb6a722df1e6fcb9662bd8d8188e1fd1873bcfb14347781c58533ad5cc361c
triton/__init__.py size=1464 sha256=08441d399eb32c7b84546a09ee497663071f04120f064390cf30e31ebea26ce3
triton/language/__init__.py size=6418 sha256=5c93d0ab5aead12a0f71faa4c3d615967ed7a8fd08db99c81b34daf9098ca626
triton/language/extra/__init__.py size=655 sha256=5d15c5bebef8d7aa51b21fd187e5faa95eba4a213254355bc69e0648013599f7
triton/language/extra/libdevice.py size=6318 sha256=0e48b5e1e95136642ccfe62dc3d0a739a2c20a7b5ee13e9c23c6cecd68cdeb70
triton/language/extra/cuda/__init__.py size=407 sha256=30106ed84518c6ca7aca08e2c0ee188755f512cc0cb2d7da8914cc48c1ad6dcc
triton/language/extra/cuda/libdevice.py size=56764 sha256=27b2a5d1e8db008bacefe6019f63922bbd65926de90bb1b527ee597477d2f365
triton/runtime/jit.py size=36766 sha256=3356ea060b9868cbb3b52de11f8f93e4223c52bbc00f2102da656ebb0295aef8
```

The narrow import/attribute probe reproduced the exact prior symptom:

```text
extra_public_names=['cuda', 'hip', 'is_pkg', 'module', 'module_finder',
                    'module_from_spec', 'module_name', 'modules', 'pkgutil', 'spec']
extra_has_libdevice=False
old_attribute_result=AttributeError: module 'triton.language.extra' has no attribute 'libdevice'
import_result=triton.language.extra.libdevice|SUCCESS|.../extra/libdevice.py
import_result=triton.language.extra.cuda.libdevice|SUCCESS|.../extra/cuda/libdevice.py
supported_has_rint=True
rint_module=triton.language.extra.cuda.libdevice
rint_signature=(arg0, _semantic=None)
rint_source_sha256=0a5eaa5b1e9f73bc8b69be11134357d714956d7589cd6ac29fb5c9e3030d85a4
torch_imported_after=False
cuda_python_modules_after=[]
```

The initiating trigger is the A4 JIT body spelling
`tl.extra.libdevice.rint`. Triton dependency discovery visits that attribute
before code generation. The masking condition is that `_kernel_bundle` imports
only `triton.language as tl`: Triton 3.4.0's `extra/__init__.py` deliberately
skips `.py` modules such as `libdevice.py` while auto-importing backend
packages, so the root stub file can exist without `libdevice` being bound on
`tl.extra`. Triton's own v3.4.0 tutorial uses
`from triton.language.extra import libdevice`; that explicit import binds the
interface module, which the CUDA backend maps to
`triton.language.extra.cuda.libdevice`. The backend implementation maps fp32
`rint` to `__nv_rintf` and fp64 to `__nv_rint`, preserving the accepted
round-to-nearest-even `rint`/CPU-reference contract. The visible symptom is the
recorded `AttributeError` during JIT dependency discovery, before codegen or
launch.

The smallest disconfirming dependency-discovery counterfactual ran three
in-memory JIT bodies under the same install and accessed each `cache_key`:

```text
initial_extra_has_libdevice=False
old_cache_key_result=AttributeError: module 'triton.language.extra' has no attribute 'libdevice'
after_supported_import_extra_has_libdevice=True
supported_cache_key_result=SUCCESS
supported_cache_key_sha256=e951d5278cfef20b5419bd2c5c5bdd75200c664ff4424ba8460b5f737afa802e1
direct_cuda_cache_key_result=SUCCESS
direct_cuda_cache_key_sha256=ac512b71869f5ef62b7a97da07bb5af85b079fd4a231ddec405131721fd8fd9e1
torch_imported=False
cuda_runtime_modules=[]
```

The supported root import then passed a compile-only SM86 front end through
TTIR, TTGIR, LLIR, PTX, and cubin in the task-local scratch cache
`/data/scratch-fast/kwen1/structured-hadamard/rot-phasea-libdevice-fix-gpu-r2/gate1-compile-cache`.
The compile emitted `supported_probe.cubin` SHA-256
`c6b270618bde4d9ab9de5b62a591329c00d64981aa3c5f9a26ea297100758283`
and PTX SHA-256
`aee3dc2cb9d940bddda7f8277ba9e423d1ac9cdb54b8b7465c802d3008e9af05`;
Torch and CUDA runtime modules remained absent.

Exact causal conclusion: this is a source-level namespace/import defect and,
therefore, a source-versus-installed-version contract mismatch, not a missing
libdevice implementation and not a GPU, allocation, visibility, CUDA math, or
scheduler defect. A version pin is disconfirmed. The chosen remedy is exactly
one explicit supported import inside the lazy Triton bundle and changing the
call to `libdevice.rint`; it keeps the task-local Triton 3.4.0 environment and
the accepted rounding semantics. Focused GPU-free regression coverage executes
the kernel body against a synthetic Triton 3.4 namespace where
`tl.extra.libdevice` is absent but the explicitly importable interface module
and its `rint` implementation are present.

### Gate 2 remedy validation

The final focused regression also asserts that the returned JIT function
closes over the explicitly imported interface module. A review-hold
counterfactual combined the new test with the exact parent implementation in a
disposable `git archive`; it failed as required with `KeyError: 'libdevice'`
because the old kernel has no such closure. The same test passes against the
fix. An earlier audit invocation accidentally ran from the source cwd and
therefore tested the fixed module instead of the archived parent; it returned
0 and was rejected as invalid evidence. The corrected invocation changed into
the disposable archive first and returned 1 on the old implementation.

The actual patched `activation_quantizer.py`, SHA-256
`eef070b590beced9772f4b58aab4f3419a66c51b8c9d499fde7f27dcffb6bee3`,
was copied to the task-local disposable CPU/static probe at
`/data/scratch-fast/kwen1/structured-hadamard/rot-phasea-libdevice-fix-gpu-r2/gate2-source-probe`.
With the exact installed Triton 3.4.0 environment, its real
`quantize_dequantize_a4_kernel` dependency cache key resolved successfully to
`445d4abeabab06335d27c2d554955f319346ad4e7b61411a7ccd03b5c4248cb853`
and compile-only SM86 codegen succeeded. The resulting cubin SHA-256 is
`2818a3a3a1745ed788b79aaa3834ea6c7eee035723ee39dc2118ac6a561526b2`,
PTX SHA-256 is
`d70058a921c031c26acdcad669af0ce39cda235853352174ed68c3f7fcdee040`,
and TTIR SHA-256 is
`ca627794fec0486c8c0fb15c2398b827bb23cb6655834573466124c1b9013e9f`.
Torch and CUDA runtime modules remained absent.

Final bounded local CPU/static validation after the regression-review
correction:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m compileall -q experiments/structured_hadamard/phase_a
  -> passed
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s experiments/structured_hadamard/phase_a/tests -v
  -> 38 tests passed in 3.370 seconds
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest experiments.structured_hadamard.phase_a.tests.test_execute -v
  -> 7 tests passed in 1.704 seconds before the closure-only test tightening;
     unaffected execution/provenance suite
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest experiments.structured_hadamard.phase_a.tests.test_stage_repository -v
  -> 3 tests passed in 0.947 seconds; unaffected staging suite
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest experiments.structured_hadamard.phase_a.tests.test_schema -v
  -> 13 tests passed in 0.089 seconds; unaffected schema suite
PYTHONDONTWRITEBYTECODE=1 python3 -m experiments.structured_hadamard.phase_a.oracle \
  --seed 0 --token-rows 2 --weight-rows 3
  -> all I/H32/H128/Hfull checks passed; maximum absolute error
     3.1086244689504383e-15; not scientific evidence
PYTHONDONTWRITEBYTECODE=1 python3 -m experiments.structured_hadamard.phase_a.preflight \
  --scheduler-clearance=false --format=jsonl \
  --result-jsonl /data/scratch-fast/kwen1/structured-hadamard/rot-phasea-libdevice-fix-gpu-r2/outputs/phase-a.jsonl
  -> exactly four I/Hfull rows; fusion=none; scientific_evidence=false
PYTHONDONTWRITEBYTECODE=1 python3 -m experiments.structured_hadamard.phase_a.preflight \
  --scheduler-clearance=false --execute
  -> refused with exit status 2
PYTHONDONTWRITEBYTECODE=1 python3 -m experiments.structured_hadamard.phase_a.execute \
  --scheduler-clearance-file /tmp/rot-phasea-libdevice-fix-gpu-r2-clearance-missing
  -> refused with exit status 2 before Torch/Triton import
git diff --check
  -> passed
```

`make` was not run because only Phase A Python, tests, and notebook provenance
changed. The diagnostic hold is resolved: implementation presence, namespace
binding, supported import, dependency discovery, actual A4 front-end codegen,
old-path test failure, fixed-path test success, rounding contract, and bounded
fail-closed suites all agree on the same source-only remedy. No GPU experiment
is needed to validate this compatibility hypothesis before staging.

The first local pre-run `git commit` invocation stopped before creating a
commit because this isolated worktree had no configured author identity. The
staged bytes were unchanged. The retry uses the exact author and committer
identity from immutable evidence commit `095cafd663` through command-local Git
environment variables, without changing shared or system Git configuration.

### Fresh r2 stage and terminal result

Status: **FAILED before SSH or Slurm.** The compatibility fix, CPU/static
validation, immutable stage, clearance, and dynamic scheduler selection all
passed. The one fresh tmux owner then exited 126 because its local owner script
was mode 0664 and was invoked as an executable. No persistent SSH route began,
no `salloc` or `srun` command ran, no job or GPU existed, and the accepted
driver was not invoked. The terminal contract forbids a replacement owner or
retry, so the task stopped.

The clean pre-run fix commit is
`03124276829a2ba8ea96166a1583b58c909dd1ff`, tree
`3c6e3772bc8ab0ecc68b46064bb107c4e22f07a1`, with exact failed-evidence
commit `095cafd6637238c53223f44d48b36b6f5211186b` and both required accepted
commits as ancestors. The full source bundle SHA-256 is
`f8984783020029ed56d304526aab568a550cddb97773b600b90e6e5b5560c23a`.
It verified from a real independent Git repository and records complete
history.

The fresh full independent stage is:

```text
/data/scratch-fast/kwen1/structured-hadamard/rot-phasea-libdevice-fix-gpu-r2/attempt-20260901T173043Z-03124276829a/stage/20260901T173043Z-03124276829a-code
```

It has an ordinary self-contained `.git`, no alternates/promisor/shallow
dependency, full connectivity, exact commit/tree/ancestry, and exactly its two
declared control files untracked. Its 1,757-entry canonical manifest SHA-256 is
`254cd9613dfea1c90c226559c2cce2f26ae26689ae362955e5406d4c82a734d1`;
manifest-file SHA-256 is
`db5c7de4090c5e14bcdd5b90b20d6541fdb1a9011a76edd8dc1ca63ba938215a`;
metadata SHA-256 is
`56f4d675adfe7b3ee83daff6bbf73b42450b271f7f0023205ed57e608b4af335`.
The staged fix, driver, and stage helper match their Git blobs, with SHA-256
values respectively
`eef070b590beced9772f4b58aab4f3419a66c51b8c9d499fde7f27dcffb6bee3`,
`a2f968b1f6efa1bc1838520802a8da9fc577ec14df5326b792c36048c6bff659`,
and `212bc2ddd82ee2f098314140e9ec62032b3dbf350f421d61a658c812a08f7a08`.

One external audit failed after the stage had already passed the driver's
verifier because the checker hard-coded obsolete `.phase-a-stage-*` control
names. The accepted names are `REPRODUCIBILITY_METADATA.json` and
`REPRODUCIBILITY_MANIFEST.json`. The failure is preserved in
`logs/stage-audit.log`; the corrected independent audit passed in
`logs/stage-audit-r2.log` without recreating or changing the stage.

The driver-validated owner-only clearance is mode 0600, UID 28131, SHA-256
`ca51742c5ec4b054dac58930d11e6523c45cbc6851d03cf9a4f0b71fdd8f4ea4`.
It binds the exact stage/commit/manifest, task-local Python path and binary
SHA-256, Triton 3.4.0 environment, absent output, `cuda:0`, fixed synthetic
`[1,11008]` seed-0 float16 workload, nearest-even A4 semantics, sequential
`fusion="none"`, 25 ms warmup, 200 ms repetition, five outer trials, model
pin `01c7f73d771dfac7d292323805ebc428287df4f9`, and Salesforce/wikitext pin
`b08601e04326c79dfdd32d625aee71d232d685c3`. Its execution-config SHA-256 is
`d9b0171c40414616dff003cd407677a4d0e71bca38b32f10d8e847a9ca6cc26c`.
The exact sole-attempt contract SHA-256 is
`4c82c54aa3b549dc302b77f22d7176e765a7e1b7c2310fcb8377af62ec8dadab`.

The read-only scheduler snapshot at `2026-09-01T17:38:08Z`, SHA-256
`4eff2bec9d637d005f2b165610f17fde4cd08623b1fa1ce4e3f74dfc5f3b4792`,
confirmed the required account and QoS. The dedicated account/QoS-allowed
Torralba capacity was: RTX 3090 12/21 free, V100 0/16 free, H100 3/8 free,
and H200 2/8 free. `vision-torralba-rtx3090` was selected without a node pin
because 24 GiB was the smallest adequate memory, V100 was full, and H100/H200
were larger and scarcer. Shared smaller-device partitions did not allow
`vision-torralba-interactive`.

Before mutation, the exact persistent owner, SSH route, `salloc` and one
`srun --pty` argv, stage, output, environment, account/QoS/partition, resources,
event token, scripts, clearance, and all-zero ledgers were recorded. The
intended request was exactly one node/task/GPU, two CPUs, 8 GiB, and ten
minutes. `sbatch`, retry, cancellation, requeue, and a second owner were not
authorized.

The registered event wait was active before tmux creation. At
`2026-09-01T17:45:22Z`, the sole tmux command used `script -qefc` to invoke
`/tmp/rot-phasea-libdevice-fix-gpu-r2-control.GOtjCa/local-owner.sh`. That file
was mode 0664 rather than executable, so the shell returned `Permission denied`
and exit 126 before the local script, its finalizer, or SSH could start. The
declared terminal event therefore could not fire. After this terminal defect
was identified, the registered wait was retired with one manual tmux signal;
it returned 1 because its declared event file was absent. No second owner was
created.

Final ledgers are:

```text
salloc_attempts=0
srun_attempts=0
driver_invocations=0
terminal_events_fired=0
terminal_events_handled=1
```

The tmux owner is absent and the registered wait is retired. There is no job
ID, partition assignment, node, CUDA visibility record, GPU identity, untimed
A4 hardware result, driver output, raw sample, or timing summary. No timing rows
were produced.

The same-device `Hfull` versus `I` sequential transform-plus-quantize overhead
and the plan's `>=5%` important-kernel gate are **not evaluated**. The identity
transform-only no-op has no denominator. This attempt establishes no RTX-3090
or other-accelerator latency/cost and makes no cross-architecture claim.

The task remains synthetic-kernel work with `scientific_evidence=false`. It
produced no model, PPL, end-to-end, Phase B/C, H32/H128-GPU, or fused-kernel
evidence. Protected job `1579631`, CNVQ owners/jobs, prior stages/owners/jobs,
other worktrees, pushes, PRs, merges, force operations, cancellations, and
requeues were untouched.

All durable attempt evidence is rooted at:

```text
/data/scratch-fast/kwen1/structured-hadamard/rot-phasea-libdevice-fix-gpu-r2/attempt-20260901T173043Z-03124276829a
```

The mode-0600 20-entry evidence manifest is `EVIDENCE_SHA256SUMS`, SHA-256
`e976828d4535acdfe870730520dc7323c5c2ed48a5e8c0103f0ef38cb3ae7d82`.
Key exact hashes are:

```text
local tmux pane log      af5d03f502a2a293d98171e2d2161e9de73b2ed23f29f02955d430a5822d9ec0
event-wait error         5d0917d60e69368677f052e209aa4b83083ba06ae32c43c6fbacfdc69f525580
terminal retirement     9acd45e402a2aa970f3c8fa9847ef6c53c3c6a3093dbcd69e5b7b6ba1391f5c3
task status             bef48445937e3ea09e29d076dbc2e307f05c49802d355f1e57b316d25685ac67
final ledgers           ba9e2b92719d9bc06a38e1033fddaf047a2ab8162428499676459bb57ea9f43f
stage creation log      d0acb9c9355ea8c69bc086589c2b1b3506c531519514ac819e337d222b558e50
corrected stage audit   13f314d2cdfc155082a6b13dc04c20cd5ef605e4fb1219ca2066b447d32fb933
scheduler snapshot      4eff2bec9d637d005f2b165610f17fde4cd08623b1fa1ce4e3f74dfc5f3b4792
```

The strict outcome is failure. The libdevice remedy is validated and retained,
but this task has no remaining retry authority and did not obtain the requested
fresh Phase A GPU evidence.

### 2026-09-01 review correction — executable libdevice regression

The Gate 2 closure/`KeyError` paragraph above is preserved as historical audit
text but is superseded as regression evidence. The final GPU-free regression no
longer inspects `co_freevars`; it invokes the synthetic executable kernel body.
With the exact parent implementation, that invocation reaches
`tl.extra.libdevice.rint` while the synthetic Triton 3.4 `extra` namespace has
no `libdevice` attribute and fails with the original recorded `AttributeError`.
With the compatibility fix, the explicit supported import binds the interface,
the invocation calls `libdevice.rint` once, and both kernel stores execute.
This correction does not change the strict failed r2 outcome or authorize a
retry.

## 2026-09-01 — Local validation recovery closeout

Status: `branch-sync-complete`. Recovery only; no corrected owner, SSH, Slurm,
CUDA, GPU, or experimental command ran.

No-mistakes run `01M1F317YT96Y6ZHVF15AKPQY6` is terminal
`completed/passed` at exact pipeline head
`9ab3c7cb76198350ee0aae29e568bbd020b27074`, tree
`0fc73c78474e764ccb108e38402b4d98b53ac709`. Its final status reports all
nine phases completed and no remaining findings. The test phase passed 47
focused Phase A `unittest` cases plus compileall, the CPU oracle, executable
parent-versus-target Triton 3.4/A4 behavior, independent staging, plan/schema,
and fail-closed execution checks. `pytest` could not start because it was not
installed in the isolated gate environment. The CI monitor observed no
registered checks and became terminal when PR #4 was closed; `passed` is the
pipeline outcome, not evidence of green remote CI checks.

Captain-authorized cleanup matched PR #4 and remote head `9ab3c7cb` exactly.
PR #4 was closed unmerged and only remote branch
`fm/structured-hadamard-phase-a-libdevice-fix-gpu-r2` was deleted. The closure
made the pipeline terminal before the approved abort was reached, so no abort
was issued. A read-only remote check then found the deleted branch absent and
the retained `refs/pull/4/head` still pinned to exact `9ab3c7cb`.

The clean local branch began recovery at
`fd00cd568e8ff63e15b2cc0213e26c9d72014b8e`, tree
`29e08ef6c8bf3c0a3238afdc7faf44735a7f2a7a`. Because guarded
`no-mistakes axi sync --recover` could not use the intentionally deleted remote
branch, the exact closed-PR head was fetched into local preservation ref
`refs/no-mistakes/recover/01M1F317YT96Y6ZHVF15AKPQY6`. Connectivity passed,
and Git proved `fd00cd568e` was an ancestor of `9ab3c7cb` with exactly nine
linear intervening commits. The checked-out research branch then advanced by
strict fast-forward, without reset, rebase, merge commit, force, discard, or
loss of the pre-recovery history.

Git also proved exact failed-run evidence commit
`095cafd6637238c53223f44d48b36b6f5211186b`, accepted Phase A commit
`d57acb60db2a4507bbff984fb3c9771e8a6ada3d`, and execution-driver commit
`21ea761c61a5e3062cea28cabda43ac04bd5278b` are all ancestors of the
validated head. This closeout entry is the only post-validation repository
change and is committed locally on top of `9ab3c7cb`; it does not alter the
validated source or authorize the corrected GPU attempt.

## 2026-09-01 — Corrected Phase A GPU owner attempt

Status: `completed`. Local-only branch
`fm/rot-phasea-libdevice-fix-gpu-r2-corrected-owner-run` began clean at
`f893845b9b91599ebd3b7a9c7f28164f39c7ed94` and advanced by strict
fast-forward to validated source/driver/transform commit
`48a972220979197359a324ab102eb8de24ce321f`, tree
`ba5f873fb9c2defa7b3af15b971cf5fb8d3092fb`. The source bundle SHA-256 is
`878cf8e836c37dced945677daaf3b1f33a3ecd38ab50f1d47c3c74a87860d5ae`.

The GPU-free Triton 3.4.0 remedy reconfirmation passed on
`slurm-login-0.csail.mit.edu`: the unsupported root attribute remained absent,
the explicit supported libdevice import exposed `rint`, the focused executable
test and real fixed-kernel dependency cache key passed, and Torch/CUDA runtime
modules remained unimported. The log SHA-256 is
`84fa6036177c14dcc94c838618eb922ebca61da691a2d9f1fa19e841d2e920d3`.

The fresh complete stage is:

```text
/data/scratch-fast/kwen1/structured-hadamard/rot-phasea-libdevice-fix-gpu-r2-corrected-owner-run/attempt-20260901T214700Z-48a972220979/stage/20260901T214700Z-48a972220979-code
```

Its ordinary independent `.git`, exact head/tree, full connectivity, 1,757
input identities, and only-two-control-file status passed independent and
driver verification. Canonical manifest, manifest-file, and metadata SHA-256
values are respectively
`e54ce37b0f14838731f6458631801e9e2130b66589e6beca27b5a1d615ce903e`,
`d5096b74326048d4ddb84607ffe43552a4372fdd715aa7ff4d1de75346c2a233`,
and `d446d626eaea2ba6b03ee0963d19e7bde87c413a8155b62c32394b97b3fc95da`.
Clearance and execution-config SHA-256 values are
`fec5407404af09cfad8eb3eabb5ff7fb5b62bc20adf92da7d45d7bc7f3c1f402`
and `d9b0171c40414616dff003cd407677a4d0e71bca38b32f10d8e847a9ca6cc26c`.

All pre-scheduler audit errors failed closed with zero ledgers and unchanged
stage/helper bytes. Preserved logs cover the initial bundle-verification cwd
error, two static-authorization snippet errors, and three launch-audit count or
`grep` errors. The final prelaunch audit passed at `2026-09-01T22:04:49Z`.
The dynamic allowed-Torralba snapshot found free 24 GiB RTX 3090 and 32 GiB
V100 capacity while H100/H200 GPUs were fully allocated, so RTX 3090 was the
smallest adequate available class. The request estimated eight active minutes
and used a 20-minute limit for a 12-minute buffer.

One tmux-owned ControlMaster, one allocation/run owner, one `salloc`, one
`srun --pty`, one scientific driver invocation, and one terminal event source
produced job `1660871`. It ran on `torralba-3090-1`, partition
`vision-torralba-rtx3090`, with one task/GPU, two CPUs, 8 GiB, account
`vision-torralba-urops-meng`, and QoS `vision-torralba-interactive`. The one
visible device was NVIDIA GeForce RTX 3090, SM86, UUID
`GPU-a8af9c30-bfc9-01c8-4e1d-f0ddbc789706`, 24,576 MiB, driver `580.178.04`,
with `CUDA_VISIBLE_DEVICES=0`, Torch `2.8.0+cu128`, CUDA 12.8, and Triton 3.4.0.
Untimed fixed A4 correctness passed before the driver. Final accounting is
`COMPLETED`, `ExitCode=0:0`, elapsed 22 seconds.

The four synchronized rows use 25 ms warmup, 200 ms repetition, and five
outer trials:

```text
transform-only I       p10=0.0000 us   median=0.0000 us   p90=0.0000 us
transform-only Hfull   p10=9.4144 us   median=10.2400 us  p90=11.2640 us
sequential T+Q I       p10=4.0960 us   median=5.1200 us   p90=5.1200 us
sequential T+Q Hfull   p10=12.5760 us  median=13.3120 us  p90=13.4080 us
```

The same-device sequential `Hfull` versus `I` overhead is
`160.0000036379789%`, so the plan's `>=5%` important-kernel threshold is met
on this exact RTX 3090 stack. All rows record `fusion="none"`;
`H32`/`H128` remain CPU/reference-only. This is synthetic non-model/non-PPL
evidence with `scientific_evidence=false` and no fused, model, PPL, end-to-end,
Phase B/C, or cross-architecture claim.

Final one-shot ledgers are `salloc=1`, `srun=1`, `driver=1`,
`terminal_fired=1`, and `terminal_handled=1`. The event was consumed, and the
route/tmux session retired at `2026-09-01T22:11:36Z`. No `sbatch`, retry,
requeue, cancellation, second owner, duplicate driver, push, PR, or merge ran;
protected job `1579631`, active CNVQ work, and all prior evidence were untouched.

Durable evidence is rooted at:

```text
/data/scratch-fast/kwen1/structured-hadamard/rot-phasea-libdevice-fix-gpu-r2-corrected-owner-run/attempt-20260901T214700Z-48a972220979
```

The 47-entry evidence manifest SHA-256 is
`f355e913a6d2230ffd10a88afee09d9794e1d2a96f0af4184e134a7e1f648020`.
Result JSONL, raw samples, execution manifest, GPU log, tmux log, terminal
accounting, and corrected result-audit SHA-256 values are respectively
`00ce50472f3690ddaffdf79cac06ab78a53b7be52d3e7f4c445f5c130084236a`,
`a4950eb79de4ac52718af0e304ee18349d41204f796f53d2418bb2845693f5c0`,
`083a216994cee6695596f255de3d22997fe3e65b0483b116ee23edd3c067fc56`,
`8d0bd0c9d33600bcf1ef6618fe9643441c92700398da641a539180dfac067e20`,
`8f31394170b07b594abe456cfdd50fc8ec0f1099f0b55579aa25b8a0b05c742b`,
`778e329615b9944b812e36ac1074a363204168c7fb00282e94f62432d73e3d6f`,
and `e19516943c61886d31418fbd3de7bba71d2ed18e14e4dc149a2bb7dfa7e69e33`.
The concise report and exact local artifact copies are under
`data/rot-phasea-libdevice-fix-gpu-r2-corrected-owner-run/`.

## 2026-09-01 — Phase A real-model W4A4 preflight

Status: `blocked-pre-scheduler`. The local-only branch
`fm/rot-phasea-real-model-baseline-r1` began clean at required source commit
`0c4aaf075d929be2474fd271ffb8bc1244206d99`, tree
`8728dfa6d4fd17027e0660c1156b6aa47700094e`. Git strict ancestry verification
passed. The result-evidence commit is recorded by the immediately following
provenance-only notebook entry.

The authoritative plan SHA-256 is
`4c0f16b28a8c92aa2a70e163df36d2e5a6e98bdcd9c4aac77a0969491f307d45`.
Its Phase A comparison rows and literal A1/A2/A3 rules were preserved. The
earlier accepted synthetic RTX 3090 kernel experiment was not rerun.

The first-priority cached model candidate was
`NousResearch/Meta-Llama-3-8B` snapshot
`315b20096dc791d381d514deb5f8bd9c8d6d3061`, selected because it is the first
complete authorized 7B-8B cache in the requested model order. Its exact config,
metadata hashes, four weight-blob identities, and WikiText-2 raw-v1 cache
revision `b08601e04326c79dfdd32d625aee71d232d685c3` are in the preflight artifact.
It was not used for a scientific row because executable W4A4 preflight failed.

The bounded installed-path audit found no packed W4A4 runtime. Int8-container
`torch._int_mm`, fp16-unpacking Triton MLIR, TorchAO A16W4/DA8W4/float8-W4,
fake quantization, and QuaRot's uninitialized timing model were explicitly
rejected as substitutes. Pinned QuaRot commit
`5008669b08c1f11f9b64d52d16fddd47ca754c5a` does contain signed packed-int4
activation and weight quantization plus a CUTLASS `int4b_t × int4b_t` GEMM with
int32 accumulation. A deterministic seed-20260901 CPU audit passed both pack
round trips, independent int32 matmul equality, and scale-dequant equality.

No compiled QuaRot or fast-Hadamard extension was present. An isolated scratch
overlay retained Python 3.10.20, Torch 2.8.0+cu128, Triton 3.4.0, Transformers
5.12.1, Datasets 4.0.0, CMake 4.1.0, and Ninja 1.13.0. A separate compiler
prefix retained CUDA NVCC 12.8.93/runtime 12.8.90, GCC/G++ 13.4.0, and sysroot
2.34. The first source build and one scratch-to-local copy entered
`rpc_wait_bit_killable` and were retired. A fresh temporary clone at the same
exact parent and submodule commits passed CMake. After the exact conda CUDA
header path corrected the first compile error, the single-worker bindings
compile remained in `mem_cgroup_handle_over_high` at 1,015,676 KiB RSS for
more than six minutes without producing the object. It was retired; all
related PIDs are gone, and no importable extension exists.

The unchanged real-model code also fails the requested support proof: QuaRot's
actual-weight checkpoint and e2e paths enumerate Llama-2 only, the cached
Llama-3 GQA configuration exposes a 512-versus-128 cache head-dimension error,
and the stock timing model is uninitialized. Moving toolchain work into the
single authorized allocation would violate the required pre-stage/runtime
boundary. No replacement kernel was implemented.

The immutable stage was therefore not created. Scheduler-query, persistent
route, tmux-owner, `salloc`, `srun`, `sbatch`, scientific-driver, retry,
requeue, cancellation, and terminal-event counts are all exactly zero. No
scheduler or protected-job state was inspected.

All required scientific rows are absent, so this is not a valid baseline. A1,
A2, and A3 are each `not evaluated`; this task cannot establish A3. The exact
local report and machine-readable preflight artifact are under
`data/rot-phasea-real-model-baseline-r1/`.

## 2026-09-01 — Phase A real-model preflight result provenance

The source commit is
`0c4aaf075d929be2474fd271ffb8bc1244206d99`. The immutable local
result-evidence commit is
`64621d4379911be6036b54e5b792a83bdd901e02`, tree
`09b7ae64942d9b69aee08027d6091ddb7a4d7955`. It contains the complete
blocked-pre-scheduler notebook entry, concise report, machine-readable
preflight artifact, and verified SHA-256 manifest. This follow-up changes only
the notebook provenance pointer.
