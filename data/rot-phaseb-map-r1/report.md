# Phase B per-layer quality–latency map

## Outcome

Phase B completed successfully on its sole authorized allocation. The literal
result is **B1: strong heterogeneity; Phase B supports proceeding to Phase C,
but Phase C was not entered**.

The result clears all three predeclared B1 gates:

- The primary sensitivity score has relative p10–p90 spread `0.731821`, above
  the `0.5` threshold.
- Rankings from deterministic calibration sequences 0–7 and 8–15 have
  Spearman `rho=0.975073`, above the `0.5` threshold.
- Replacing `Hfull` with `H32` at the frozen three predicted-most and three
  predicted-least sensitive sites passes the predeclared qualitative PPL gate:
  top-three mean impact `1.165429` PPL is above bottom-three mean impact
  `0.548384`, and the six-site Spearman correlation is positive at `0.257143`.
- Median affected packed-W4A4 layer latency is `0.731136 ms` for `Hfull` and
  `0.452608 ms` for `H32`. `Hfull` is `61.5385%` slower, well above the `5%`
  separation threshold.

The primary proxy therefore remained active. The predeclared stronger proxy
was not invoked; `switch_count=0`. This is evidence for a Phase C selector,
not Phase C evidence and not a fused-kernel result.

## Scope and method

The run evaluated exactly the 32 Llama-3 FFN `down_proj` input sites and the
four choices `I`, `H32`, `H128`, and `Hfull`. It did not add attention sites,
KV-cache transforms, learned transforms, lower bits, another model, or Phase C.

Calibration used exactly 16 WikiText-2 train sequences by 512 tokens. Each
site retained 8,192 fp16 activation rows and its fp16 reference output in one
301,991,909-byte cache object. The driver processed and released one site at a
time. The 32 immutable cache objects total 9,663,741,088 bytes. The sample was
not expanded.

The deterministic stability split was frozen before the result:

- subset A: sequence indices 0–7, rows 0–4,095;
- subset B: sequence indices 8–15, rows 4,096–8,191;
- statistic: Spearman rank correlation of the 32 per-site sensitivity scores;
- stable threshold: `rho >= 0.5`.

The primary score was `NMSE(H32) - NMSE(Hfull)`. The three largest and three
smallest scores were frozen before any PPL result was viewed. Each B3 variant
then changed one selected site from `Hfull` to `H32`, held all other 31
down-projection sites at `Hfull`, and reused the accepted Phase A evaluation.

Every transform and packed GEMM was sequential with `fusion="none"`. Timing
used the exact shape `[1, 14336]`, 20 warmups, 100 repetitions, CUDA events,
and a terminal `torch.cuda.synchronize`. The result contains 128 machine rows
and 25,600 raw timing samples.

## Transform semantics and correctness

`H32` and `H128` are real normalized Sylvester block-Hadamard transforms, not
labels or aliases. At width 14,336, `H32` uses 448 consecutive blocks and
`H128` uses 112 consecutive blocks. The same symmetric orthonormal transform
is folded into each matching weight row. `Hfull` reuses the accepted QuaRot
`7 x 2048` factorization, while `I` remains the comparable identity alias.

The deterministic inverse and folded-weight checks passed at tolerance
`1e-12`:

| Transform | Inverse max abs | Folded equivalence max abs |
|---|---:|---:|
| `H32` | `4.44e-16` | `1.78e-15` |
| `H128` | `4.44e-16` | `1.33e-15` |
| `Hfull` | `0` | `0` |

The apparent values above are below the configured tolerance. No CPU-only
surrogate or full-Hadamard alias was used for either block transform.

## Quality–latency map

The table reports medians across the 32 site-level values. Local quality is
normalized output error after the accepted packed W4A4 runtime.

| Choice | Median local NMSE | Median activation max | Median transform ms | Median affected-layer ms |
|---|---:|---:|---:|---:|
| `I` | `0.491805` | `9.21484` | `0` | `0.178176` |
| `H32` | `0.128814` | `1.67529` | `0.293888` | `0.452608` |
| `H128` | `0.079647` | `0.919189` | `0.376552` | `0.537600` |
| `Hfull` | `0.058091` | `0.354858` | `0.572552` | `0.731136` |

Median activation RMS is approximately `0.05613` for every choice, as expected
for orthonormal transforms; the max statistic captures the outlier reduction.
The full 128-row table and figure are
[`phase-b-table.csv`](artifacts/run-1662528/scientific-run/phase-b-table.csv)
and
[`phase-b-quality-latency-map.svg`](artifacts/run-1662528/scientific-run/phase-b-quality-latency-map.svg).

The active sensitivity scores range from `0.000530` to `0.134249`. The frozen
most-sensitive sites were layers 27, 2, and 30. The frozen least-sensitive
sites were layers 10, 11, and 1.

## Six-site WikiText-2 validation

The accepted Phase A all-`Hfull` reference PPL is `101.042258`. Each row below
is a single-site `Hfull -> H32` replacement, with every other setting fixed.

| Predicted group | Layer | Proxy score | PPL | Impact vs Hfull |
|---|---:|---:|---:|---:|
| most sensitive | 27 | `0.134249` | `101.094260` | `+0.052002` |
| most sensitive | 2 | `0.116702` | `103.445043` | `+2.402785` |
| most sensitive | 30 | `0.093713` | `102.083760` | `+1.041502` |
| least sensitive | 10 | `0.034443` | `102.073577` | `+1.031319` |
| least sensitive | 11 | `0.033535` | `102.350227` | `+1.307969` |
| least sensitive | 1 | `0.000530` | `100.348124` | `-0.694135` |

The primary proxy passes the rule declared before PPL, but its six-site
correlation is modest. Layer 27 has almost no measured PPL impact, while layers
10 and 11 have impacts near or above one point. B1 should therefore be read as
“actionable under the predeclared qualitative gate,” not as evidence of a
high-fidelity site-by-site PPL predictor. The negative layer-1 impact is also a
single controlled measurement, not proof that H32 improves quality there in
general.

Absolute quality remains a major limitation inherited from Phase A. The
minimal down-projection-only packed-W4A4 `Hfull` baseline has PPL `101.042258`
versus FP16 PPL `6.820283`. Phase B establishes useful relative heterogeneity
inside this accepted test bed; it does not establish deployment-quality W4A4.

## Literal Decision B

- **B1 applies.** Quality sensitivity is heterogeneous and stable, the frozen
  proxy passes its PPL check, and realized latency differs by more than 5%.
- **B2 does not apply.** Realized `H32` and `Hfull` latency is not nearly
  identical.
- **B3 does not apply.** Actionable quality heterogeneity was established.
- **B4 does not apply.** Both quality heterogeneity and latency separation were
  established.

The authoritative plan directs B1 to Phase C. This task deliberately stopped
at the Phase B boundary.

## Provenance and exact inputs

The authoritative plan SHA-256 was
`4c0f16b28a8c92aa2a70e163df36d2e5a6e98bdcd9c4aac77a0969491f307d45`.
The accepted Phase A report SHA-256 was
`41a081045f29f763829000f66283f6f818928573abf20aa72f7d1eae6ba482de`.

The clean branch `fm/rot-phaseb-map-r1` was moved from `origin/main` to the
Phase A provenance tip only through the mandated
`git rebase --onto 3bc279eca8 origin/main`. The scientific source commit was
`621e35b5d890ea0623f232a8aada299a1fc30d97`, tree
`3f47e8e3332447b9d4f5c61b561cceca3868ccbd`. The immutable full-repository
stage had independent Git metadata, 1,838 entries, and canonical entries hash
`7bc839e09dcfda3c142ac347c85e903690fd7740fcd9af8a9138c36186135a92`.
Its metadata and manifest-file hashes were `6c663f655271065c59e1c2e6bf541c6f4c54269d0f468eacc282278c25a2e9cd`
and `bbd8f72fd8957af0c81425f339697af8e0017b6c2a403fab533b740be236ee99`.

Pinned scientific inputs were:

- model: `NousResearch/Meta-Llama-3-8B` revision
  `315b20096dc791d381d514deb5f8bd9c8d6d3061`;
- dataset: `Salesforce/wikitext` revision
  `b08601e04326c79dfdd32d625aee71d232d685c3`;
- calibration Arrow SHA-256
  `57947bc7b58df4b19662c0609cc30651bc84328dab5fd588860b752072911789`;
- evaluation Arrow SHA-256
  `2b8a3efac7b468cbe6432edba5f55c21e435d93873acc6727431f08d5ed328ea`;
- accepted packed-W4A4 runtime SHA-256
  `10a961e8855d7349708412aa8c21a38667fb75c94c372180dd6f992913e0a8e0`;
- QuaRot commit `5008669b08c1f11f9b64d52d16fddd47ca754c5a`;
- CUTLASS commit `ffa34e70756b0bc744e1dfcc115b5a991a68f132`;
- Python 3.10.20 SHA-256
  `fa10ee8f4c18e62cbd1e467c156a228be45138bd537b9949e66fc8e5937a018e`;
- exact 147-distribution inventory SHA-256
  `6acfbf19d91f7c2ad81fa9702759940896860f8aa821c32cbf6b4c6190b5aaa2`.

The staged audit passed before allocation, binding the project and dependency
manifests, model and data bytes, extension target, helper hashes, package
bytes, exact argv, output parent, and zeroed ledger. Clearance SHA-256 was
`ecc98c92ed9c26644713ab66a4ba8805c31e079b21d968a97cdcb03f71e2dd3c`;
preflight result SHA-256 was
`a7c43107febef65bce893e236a1fc3bb546d10cff2db60de026b1262248b17ae`.

## Scheduler, hardware, and terminal lifecycle

Read-only account and capacity inspection selected
`vision-torralba-rtx3090`, the smallest available account-allowed class with a
native SM86 image in the accepted extension. This is the same GPU class and
UUID as Phase A, so the latency comparison remains direct. No PTX-JIT fallback
was used.

Slurm job `1662528` ran on `torralba-3090-1.csail.mit.edu` with one NVIDIA
GeForce RTX 3090, compute capability 8.6, UUID
`76f2cc49-6a31-24e3-23e0-f628fc728221`, and 25,296,044,032 visible bytes.
Accounting recorded `COMPLETED`, `ExitCode=0:0`, and `00:14:39` elapsed. The
step MaxRSS was `33,546,248K` against requested `33,554,432K`, or `99.975610%`.
That confirms the decision not to expand the calibration sample. The result's
driver field is `null` because the pinned Torch version lacks the private query
used by the driver; no second run was made to repair a metadata-only gap.

The fixed terminal event was armed before the scheduler boundary. Exactly one
owner, `salloc`, `srun --pty`, scientific driver, and terminal event ran. The
final ledger is:

```json
{
  "driver_attempts": 1,
  "owner_attempts": 1,
  "salloc_attempts": 1,
  "srun_attempts": 1,
  "terminal_events_fired": 1
}
```

The event was handled once, result and accounting bytes were preserved, and
the GPU owner, SSH control master, and sole tmux route were retired. There was
no `sbatch`, retry, requeue, cancellation, second stage, second owner, duplicate
driver, push, PR, merge, protected-job query, or active-CNVQ contact.

## Preserved corrections

### Malformed read-only remote metadata command

Before the supervision correction, one read-only metadata probe was attempted
as this local command:

```text
/usr/bin/ssh -S /tmp/phaseb-map-r1.szMkNt/ssh -o ControlMaster=no -o BatchMode=yes slurm-login.csail.mit.edu /data/scratch-fast/kwen1/micromamba/root/envs/causal_forcing/bin/python3.10 -c 'import json; from pathlib import Path; r=Path("/data/scratch-fast/kwen1/structured-hadamard/rot-phaseb-map-r1/attempt-20260902T022201Z-621e35b5d8/stages/project"); m=json.loads((r/"REPRODUCIBILITY_METADATA.json").read_text()); print("remote_head="+m["source_head"]); print("remote_entries="+str(m["tracked_and_untracked_entries"])); print("remote_entries_sha256="+m["working_tree_manifest_sha256"])'
```

SSH command serialization stripped the intended grouping, so the remote shell
saw `python3.10 -c import json; ...` and returned a syntax error at `(`. The
failure was masked only in the sense that it inspected no metadata at all; it
was not treated as evidence. It was read-only, occurred after the sole stage
had completed, and changed no stage byte.

The correction copied the already-generated
`REPRODUCIBILITY_METADATA.json` bytes with `scp`, verified SHA-256
`6c663f655271065c59e1c2e6bf541c6f4c54269d0f468eacc282278c25a2e9cd`,
and parsed them locally with `jq`. All later remote actions used committed or
staged scripts, copied metadata bytes, or ordinary non-Python commands. No
further remote inline-Python or heredoc probe was constructed.

The pre-correction issue line in
`/home/ubuntu/firstmate/state/structured-rotation-secondmate.status` remains as
evidence and was not deleted or rewritten. After the supervision correction,
this worker directly targeted only
`/home/ubuntu/.treehouse/firstmate-557e63/5/firstmate/state/rot-phaseb-map-r1.status`.
The old parent ledger was not directly targeted again by this worker.

### Accounting local-time correction

The first post-terminal `sacct` command supplied the UTC date
`2026-09-02T00:00:00` to Slurm's local-time parser. Slurm rejected it as later
than the local end time and returned no rows. The command and error are
preserved in `accounting-first-command.txt`. One corrected read used
`2026-09-01T00:00:00` and produced `sacct.psv`; this was accounting
preservation, not a scheduler or GPU retry.

## Validation and artifacts

Before staging, 13 focused Phase B unit tests and all 60 inherited Phase A
tests passed. Direct CLI import, compileall, and `git diff --check` passed.
`pytest -s --tb=short` could not start because `pytest` was absent from the
local environment. `make` was not run because the changes were Python,
documentation, and data only.

After terminal, committed verifier code independently checked the 128-row
schema, 25,600 raw timing keys, artifact hashes, selection freeze, six-site
agreement, literal decision, one-shot ledger, terminal record, and all 32 cache
objects. It passed with `site_cache_hashes_verified=true`.

Key artifact hashes are:

```text
results.json                      055f180acf4a69023cf1352c9d9605ce9e4b116156fdad180e8eed4af7a4d118
map-rows.jsonl                    b5461aa9af17a85936f55c57c9c55d754a774236313c41363b62b6f849d8ef62
raw-timings.jsonl                 9f5099b92add5cc9434ca7ecc2cf5d3866531702a42bce264d2ca354c538eb6d
selection-freeze.json             7c7ef11ba7eded7082849aa5a0fd98928ffb3e44c984005adf926dc7a15de166
six-site-validation.jsonl         957b66a00b811b4434038509abdb10bd890f5d963671277732676ebe68952458
ranking-stability-analysis.json   a48f1259c26673fdc39db30c071bc6cd54b4788871d9856b3a95d745452764ce
site-cache-manifest.json          5cd321e34298b572dcf689e866a9074436e93d394ebe9fb81e313eb4a444ba7f
phase-b-quality-latency-map.svg   3decd7844bae57f720607598045be51c8e40a294d40a83f5e46169a2902572c6
artifact-manifest.sha256          47379c6db422036826eebaf733d8f76c60b7cbffd4abc337e5ec4f69012f6f50
```

The complete durable remote result, including the 9.1 GiB cache, is under:

```text
/data/scratch-fast/kwen1/structured-hadamard/rot-phaseb-map-r1/attempt-20260902T022201Z-621e35b5d8
```

The concise branch evidence is under
`data/rot-phaseb-map-r1/artifacts/run-1662528/`. Exact separated command arrays
are in `provenance/execution-argv.json`, SHA-256
`dee56cf11ca9a45934a9d4cd45c45e6aedafdf57060fd0d20d7d4a58506731d4`;
their canonical array hash is
`4c53ea10b99e7bf1e2ae502cdea28780b91317d5ab65781871432ec74ca4ef31`.

To verify the copied result without a GPU:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  experiments/structured_hadamard/phase_b/verify_results.py \
  data/rot-phaseb-map-r1/artifacts/run-1662528
```

The large-cache verification additionally accepts
`--cache-root <remote-or-local-site-cache-directory>`. Reproduction of the GPU
experiment must use the preserved stage, clearance, argv, and one-shot owner
contract; this completed task does not authorize another attempt.

## Completion-gate inventory

The report exposes no unresolved choice that belongs to the captain. B1's next
direction is already fixed by the authoritative experimental plan, while this
task's boundary explicitly forbids entering Phase C. The shared
`decision-hold-lifecycle` gate is therefore attested with `--none`; no dependent
work was created in this task.
