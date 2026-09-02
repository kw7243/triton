# Phase C five-budget Pareto selector

## Outcome

Phase C completed on its sole authorized allocation. The literal result is
**C4: the adaptive allocation does not beat the fixed-transform frontier; stop
the per-layer allocation idea and do not add selector search complexity**.

The three interior adaptive points gained `21.0787%`, `13.5370%`, and
`6.7512%` decode speed over all-`Hfull`, but their WikiText-2 PPL penalties were
`+403.0133`, `+119.2861`, and `+21.6547`. None stayed within the frozen
`0.2`-PPL quality window. The adaptive curve also did not Pareto-dominate the
fixed choices over multiple budgets.

This is a relative result in the inherited test bed. All-`Hfull` packed W4A4
PPL is `101.042258`, versus inherited FP16 PPL about `6.82`; Phase C does not
establish deployment-quality quantization.

## Frozen selector

The policy was fixed before any Phase C PPL or decode timing was visible. It
uses only the accepted Phase B map for the 32 Llama-3 FFN `down_proj` inputs
and the real `I`, `H32`, `H128`, and `Hfull` implementations.

Starting from all-`Hfull`, each layer can move only through adjacent downgrades
`Hfull -> H128 -> H32 -> I`. At each step, the selector computes incremental
predicted local-NMSE loss divided by measured transform latency saved and
chooses the smallest tuple `(rho, layer index, target-transform rank)`. Strictly
positive saved latency is required. Each budget chooses the earliest nested
trace prefix at or below its target, which makes assignments and realized cost
monotone. The full trace has 96 moves.

The all-`Hfull` predicted transform cost is `18.38572797179222 ms`. The frozen
targets and realized costs are:

| Policy | Target | Realized cost | Realized share |
|---|---:|---:|---:|
| adaptive 0% | 0% | 0 ms | 0% |
| adaptive 25% | 25% | 4.405312 ms | 23.960498% |
| adaptive 50% | 50% | 9.177488 ms | 49.916370% |
| adaptive 75% | 75% | 13.647504 ms | 74.228793% |
| adaptive 100% | 100% | 18.385728 ms | 100% |

Every frozen assignment carries its 32 exact Phase B source-row hashes. The
five adaptive identities plus four fixed baselines deduplicate to seven unique
measurements: adaptive 0% is fixed `I`, and adaptive 100% is fixed `Hfull`.
The complete freeze is
[`policy-freeze.json`](policy-freeze.json), SHA-256
`2543c2ed55a1d0b9b70189d53c91c27b88c5db4a15b152786cdbcb1b4d59a31a`.

No pinned, already-reproducible comparable MixQuant/PeRQ implementation exists
in the accepted repository or dependency set. It is explicitly unavailable;
no approximation or reproduction was attempted.

## Five-point Pareto result

All policies used WikiText-2 test PPL and batch-1 decode with 128 prompt tokens,
32 output tokens, one warmup, five repetitions, seed `20260902`, explicit CUDA
synchronization, the accepted sequential signed-W4A4 transform-plus-quantize
semantics, and `fusion="none"`.

| Adaptive budget | Realized share | PPL | Median decode ms | Tokens/s | Speedup vs `Hfull` |
|---:|---:|---:|---:|---:|---:|
| 0% | 0% | 1075.362051 | 1701.090 | 18.8115 | 22.3906% |
| 25% | 23.960498% | 504.055520 | 1810.279 | 17.6768 | 21.0787% |
| 50% | 49.916370% | 220.328343 | 1930.525 | 16.5758 | 13.5370% |
| 75% | 74.228793% | 122.696919 | 2053.243 | 15.5851 | 6.7512% |
| 100% | 100% | 101.042258 | 2191.861 | 14.5995 | 0% |

The 0% speedup shown above is calculated from the measured endpoint for
context; endpoint decisions were deduplicated exactly as frozen.

The fixed baselines were:

| Fixed transform | Realized share | PPL | Median decode ms | Tokens/s |
|---|---:|---:|---:|---:|
| `I` | 0% | 1075.362051 | 1701.090 | 18.8115 |
| `H32` | 51.301009% | 179.977336 | 1941.881 | 16.4789 |
| `H128` | 65.621508% | 123.946654 | 2009.515 | 15.9242 |
| `Hfull` | 100% | 101.042258 | 2191.861 | 14.5995 |

At roughly half cost, adaptive 50% is only `0.5848%` faster than fixed `H32`
but is `40.3510` PPL worse, so it misses the frozen material-latency threshold
and loses badly on quality. Adaptive 75% improves PPL by `1.2497` relative to
fixed `H128`, but is `2.1760%` slower. These are tradeoff points, not adaptive
dominance. No adaptive interior policy materially dominates any fixed block
baseline, and no fixed baseline is dominated across multiple adaptive budgets.

The main machine-readable table is
[`phase-c-pareto-table.csv`](artifacts/job-1663316/scientific-run/phase-c-pareto-table.csv)
and the frozen five-point figure is
[`phase-c-five-budget-pareto.svg`](artifacts/job-1663316/scientific-run/phase-c-five-budget-pareto.svg).
Raw timings, exact assignments, measurements, and the literal decision are in
the same `scientific-run` directory.

## Literal Decision C

- C1 does not apply: no adaptive point achieves at least 3% speedup within
  `0.2` PPL, and the adaptive policy does not dominate fixed baselines over
  multiple budgets.
- C2 does not apply: there is no quality-good 1–3% speedup point.
- C3 does not apply under the declared precedence: speed improves as quality
  degrades, but the selector first fails the fixed-frontier test.
- C4 applies: stop the per-layer allocation idea. Do not add selector search
  complexity.

No C-kernel branch, mass-balancing recovery, Phase D, accuracy recovery,
MixQuant/PeRQ reproduction, fusion branch, or second scientific run was
entered. The plan owns any automatic post-C4 routing; this task adds no new
authority request.

## Provenance and frozen inputs

The authoritative plan SHA-256 is
`4c0f16b28a8c92aa2a70e163df36d2e5a6e98bdcd9c4aac77a0969491f307d45`.
The accepted Phase B report SHA-256 is
`896d74da1e00d56bb260bd39a56d0cc42e66b3554916a67c808b8b9c44d53f87`,
and its 128-row map SHA-256 is
`b5461aa9af17a85936f55c57c9c55d754a774236313c41363b62b6f849d8ef62`.

The clean zero-unique branch `fm/rot-phasec-selector-r1` was moved from
`origin/main` to accepted Phase B tip
`f7cd759307ffdb534581587520b42e95a309498c` only with the required
`git rebase --onto`. Scientific source commit
`94d23efadda38451dabd120eaa68ec6f133a369a`, tree
`cd4a2eeb2e8a13178f77af0aba9d73bd0ef54fa6`, contains only the frozen,
outcome-free implementation and contract.

The sole complete immutable stage is:

```text
/data/scratch-fast/kwen1/structured-hadamard/rot-phasec-selector-r1/
  attempt-20260902T034239Z-94d23efadd/stages/project
```

It has independent usable Git metadata, exact scientific source, 1,885
tracked/untracked/allowed-ignored entries, and canonical entries hash
`af221f7e29c3cad6cde62416afe32d928e2a648aaa679be7e20cccd97a73e8c3`.
Stage metadata SHA-256 is
`428c991f68cc1fe24f535d4de2cb49bbb84f5d6b3e29dcaa8d54ca57cf451850`;
the manifest-file SHA-256 is
`c1045554ff5b7b5e3f86e2aafb416a7722bc94ceec9af12da1d3f6eff09c6112`.
The complete source bundle SHA-256 is
`c48d0ef48306663b56e2d52c4c0e045e3334941c222d65d11c3949835982aaf2`.

The staged preflight bound the plan, Phase B report/map/results and all 224
selected source rows, model/data bytes, QuaRot/CUTLASS commits, native runtime,
exact environment/package bytes, helper bytes, frozen policy, argv, output
parent, and zero ledger. Clearance SHA-256 is
`503f34523a6a49146ef31d6e9d98aef795552e948c81c260546d9af002397c31`;
preflight SHA-256 is
`8f628f7508075e899d98f1d142887efcb80ec5ddacc0a451e4fe8a39874a3cc7`.

Pinned scientific inputs are:

- `NousResearch/Meta-Llama-3-8B` revision
  `315b20096dc791d381d514deb5f8bd9c8d6d3061`;
- `Salesforce/wikitext` revision
  `b08601e04326c79dfdd32d625aee71d232d685c3`, evaluation Arrow SHA-256
  `2b8a3efac7b468cbe6432edba5f55c21e435d93873acc6727431f08d5ed328ea`;
- accepted packed-W4A4 runtime SHA-256
  `10a961e8855d7349708412aa8c21a38667fb75c94c372180dd6f992913e0a8e0`,
  with native SM80 and SM86 SASS;
- QuaRot commit `5008669b08c1f11f9b64d52d16fddd47ca754c5a` and CUTLASS commit
  `ffa34e70756b0bc744e1dfcc115b5a991a68f132`;
- Python 3.10.20 SHA-256
  `fa10ee8f4c18e62cbd1e467c156a228be45138bd537b9949e66fc8e5937a018e`
  and exact 147-distribution inventory SHA-256
  `6acfbf19d91f7c2ad81fa9702759940896860f8aa821c32cbf6b4c6190b5aaa2`.

Exact owner, `salloc`, allocated owner, `srun --pty`, and driver arrays are
preserved in
[`execution-argv.json`](artifacts/job-1663316/provenance/execution-argv.json),
file SHA-256
`6ac4d36e793423d0ec95e9d695b9f5ab01c7a50f1511e57ebf73dd38939a05a9`;
their canonical arrays hash is
`c3011c3d8241503cfe1ab31792191f8ab1e6e7a3b3c3c923f7f32f01433d4621`.

## Scheduler, hardware, and lifecycle

Read-only inspection immediately before launch found eight free RTX 3090 GPUs
and selected `vision-torralba-rtx3090`, the smallest available account-allowed
class with a native accepted SM86 target. Job `1663316` ran on
`torralba-3090-1.csail.mit.edu`, NVIDIA GeForce RTX 3090, SM86, UUID
`76f2cc49-6a31-24e3-23e0-f628fc728221`, with 25,296,044,032 visible bytes.
This is the accepted Phase A/B GPU class and UUID, so comparison is direct.

The request was one node, one GPU, two CPUs, 32 GiB, and 45 minutes. Accounting
is `COMPLETED`, exit `0:0`, elapsed `00:12:49`; the scientific step used
`30,851,108K` MaxRSS, `91.943467%` of 32 GiB. Phase C evaluated policies
sequentially, released each model, and did not recreate the 9.1 GiB Phase B
cache.

The fixed terminal event was armed before the scheduler boundary. Exactly one
owner, `salloc`, `srun --pty`, driver, and event ran:

```json
{
  "driver_attempts": 1,
  "owner_attempts": 1,
  "salloc_attempts": 1,
  "srun_attempts": 1,
  "terminal_events_fired": 1
}
```

The event was handled once. Results, logs, hashes, terminal state, and
accounting were preserved; the allocation was relinquished and the sole SSH
control master/tmux route retired. There was no `sbatch`, retry, requeue,
cancellation, duplicate stage/owner/driver, PTX fallback, protected-job query,
active-CNVQ contact, push, PR, merge, CI, or no-mistakes pipeline.

## Preserved read-only corrections

Three harmless command-shaping issues are retained in task status:

1. A pre-stage remote `stat` format containing spaces was split by SSH
   serialization. Its partial output was rejected; a no-space format token was
   used instead. No remote byte or one-shot ledger changed.
2. One capacity probe gave three node names to a single `scontrol show node`
   invocation, which Slurm rejected. Separate read-only node queries supplied
   the capacity evidence. No job or allocation existed.
3. After terminal, a `find -printf` format containing `|` was split by SSH
   serialization and produced no usable listing. The already-terminal fixed
   attempt was then copied with shell-neutral paths. No scheduler/GPU action
   occurred.

None was treated as scientific evidence, and none caused a second stage or
scientific attempt.

## Validation and reproduction

Before launch, direct local validation passed `compileall`, all 60 inherited
Phase A tests, all 13 inherited Phase B tests, 8 focused Phase C tests, the
accepted Phase B verifier, CLI import/help checks, and `git diff --check`.
Pytest was unavailable. `make` was correctly omitted because changes are only
Python, documentation, and data. The intake correction was honored: no
no-mistakes/CI/push/PR pipeline was created or resumed.

Independent post-terminal verification reproduced the freeze, seven measured
schemas, 35 raw timing keys, assignment aliases, every declared artifact hash,
literal C4, passed preflight, and exact one-shot ledger. It is preserved in
[`result-verification.json`](artifacts/job-1663316/result-verification.json),
SHA-256
`50359aa1e6a377cf7cb4d5e5640a1dde14e3ea8b9727156023e0a7dd511e418d`.
The 21-file evidence hash list is
[`artifact-sha256.txt`](artifacts/job-1663316/artifact-sha256.txt), SHA-256
`af0e454600913dab592ad54834164d487020549db87514e2436dbe0c46c73e98`.

From the repository root, the copied evidence can be checked without a GPU:

```bash
sha256sum -c data/rot-phasec-selector-r1/artifacts/job-1663316/artifact-sha256.txt
PYTHONDONTWRITEBYTECODE=1 python3 -m \
  experiments.structured_hadamard.phase_c.verify_results \
  data/rot-phasec-selector-r1/artifacts/job-1663316 \
  --policy-freeze data/rot-phasec-selector-r1/policy-freeze.json
```

The complete immutable remote attempt remains under the `kwen1` scratch path
above. Re-executing the scientific argv would be a second GPU attempt and is
not authorized by this completed task.
