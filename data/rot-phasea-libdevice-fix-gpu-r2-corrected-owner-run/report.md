# Corrected Phase A owner attempt

Status: **completed successfully.** Job `1660871` produced the four accepted
`I`/`Hfull` timing rows on one NVIDIA GeForce RTX 3090. The same-device
sequential transform-plus-quantize median increased from `5.120 us` for `I` to
`13.312 us` for `Hfull`, an overhead of `160.000%`. The accepted `>=5%`
important-kernel threshold is met on this exact RTX 3090 stack.

This is synthetic fixed-seed kernel evidence with
`scientific_evidence=false`. It is not model, perplexity, end-to-end, Phase
B/C, H32/H128-GPU, or fused-kernel evidence.

## Timing result

All values are microseconds from synchronized CUDA device events. The
configuration was 25 ms warmup, 200 ms repetition, and five outer trials.
`I` transform-only is the specified host no-op and is not used as a latency
denominator.

| Timing boundary | Transform | p10 (us) | Median (us) | p90 (us) | Launches |
|---|---|---:|---:|---:|---:|
| transform-only | `I` | 0.0000 | 0.0000 | 0.0000 | 0 |
| transform-only | `Hfull` | 9.4144 | 10.2400 | 11.2640 | 2 |
| sequential transform-plus-quantize | `I` | 4.0960 | 5.1200 | 5.1200 | 1 |
| sequential transform-plus-quantize | `Hfull` | 12.5760 | 13.3120 | 13.4080 | 3 |

The comparable-row calculation is:

```text
(13.311999849975109 / 5.119999870657921 - 1) * 100
= 160.0000036379789%
```

Every row records `fusion="none"`. The combined boundary is sequential
transform-then-quantize and makes no fused-kernel claim. `H32` and `H128`
were used only by CPU/reference checks and do not appear as measured GPU rows.

The independent post-run audit passed the schema, exact four-row matrix,
output-manifest digests, raw-sample finiteness/completeness, recomputed
quantiles, timing configuration, synchronization, correctness, workload,
fusion, and evidence-label checks. Raw sample counts were 2,633 for `Hfull`
transform-only, 3,238 for `I` transform-plus-quantize, and 3,160 for `Hfull`
transform-plus-quantize; the `I` host no-op correctly has no samples.

## Correctness and hardware

- Job/allocation: `1660871`, terminal `COMPLETED`, exit `0:0`, elapsed 22 s
- Partition/node: `vision-torralba-rtx3090` / `torralba-3090-1`
- Account/QoS: `vision-torralba-urops-meng` /
  `vision-torralba-interactive`
- Resources: one node, one task, one GPU, two CPUs, 8 GiB RAM, 20-minute limit
- GPU: NVIDIA GeForce RTX 3090, Ampere/SM86, compute capability 8.6
- UUID: `GPU-a8af9c30-bfc9-01c8-4e1d-f0ddbc789706`
- VRAM/driver: 24,576 MiB / `580.178.04`
- Visibility: `CUDA_VISIBLE_DEVICES=0`; Torch saw exactly one CUDA device
- Runtime: Python 3.10.20, Torch `2.8.0+cu128`, CUDA 12.8, Triton 3.4.0

The pre-driver gate synchronized a real CUDA tensor operation and passed the
corrected nearest-even A4 path against the CPU reference: relative error
`0.00022564938232343555`, maximum absolute error
`0.0008370535714283811`. The accepted driver then passed its independent full
CPU oracle plus GPU `I`/`Hfull` transform and quantizer checks. The maximum
recorded `Hfull` local-equivalence relative error was
`0.00020514586073842105`; maximum absolute error was
`0.000974272670484666`.

The accelerator is the original RTX 3090 target, so this run is direct evidence
for that same-device candidate gate. Absolute latency and hardware-specific
conclusions remain tied to this exact model, driver, runtime, and clock policy;
they do not transfer to other GPU classes without validation there.

## Source, stage, and environment

- Local-only branch:
  `fm/rot-phasea-libdevice-fix-gpu-r2-corrected-owner-run`
- Validated source/driver/transform commit:
  `48a972220979197359a324ab102eb8de24ce321f`
- Source tree: `ba5f873fb9c2defa7b3af15b971cf5fb8d3092fb`
- Source bundle SHA-256:
  `878cf8e836c37dced945677daaf3b1f33a3ecd38ab50f1d47c3c74a87860d5ae`
- Attempt root:
  `/data/scratch-fast/kwen1/structured-hadamard/rot-phasea-libdevice-fix-gpu-r2-corrected-owner-run/attempt-20260901T214700Z-48a972220979`
- Stage:
  `/data/scratch-fast/kwen1/structured-hadamard/rot-phasea-libdevice-fix-gpu-r2-corrected-owner-run/attempt-20260901T214700Z-48a972220979/stage/20260901T214700Z-48a972220979-code`
- Canonical 1,757-entry manifest SHA-256:
  `e54ce37b0f14838731f6458631801e9e2130b66589e6beca27b5a1d615ce903e`
- Manifest-file SHA-256:
  `d5096b74326048d4ddb84607ffe43552a4372fdd715aa7ff4d1de75346c2a233`
- Metadata SHA-256:
  `d446d626eaea2ba6b03ee0963d19e7bde87c413a8155b62c32394b97b3fc95da`
- Clearance SHA-256:
  `fec5407404af09cfad8eb3eabb5ff7fb5b62bc20adf92da7d45d7bc7f3c1f402`
- Execution-config SHA-256:
  `d9b0171c40414616dff003cd407677a4d0e71bca38b32f10d8e847a9ca6cc26c`

The clean local task branch advanced from `f893845b9b91599ebd3b7a9c7f28164f39c7ed94`
to the validated head by strict fast-forward. Staging began from an exact clean
standalone remote clone with 1,757 tracked inputs and no included dirty,
untracked, or allowed-ignored inputs. The resulting stage has ordinary
self-contained `.git` metadata, no alternates/promisor/shallow dependency,
full connectivity, exact head/tree, manifest-bound bytes, and only its two
declared control files untracked. Both the independent audit and the driver's
static verifier passed before scheduler mutation.

The GPU-free Triton 3.4 reconfirmation also passed before staging: the old
`tl.extra.libdevice` attribute was absent, the supported explicit
`from triton.language.extra import libdevice` bound `rint`, the focused
executable regression passed, the fixed kernel dependency cache key resolved,
and neither Torch nor a CUDA runtime module was imported. The exact environment
package list and redacted allocated environment are preserved in `artifacts/`.

## Scheduler selection and ownership

The read-only association showed the required account and interactive QoS.
The dedicated allowed Torralba snapshot found four GPU partitions. RTX 3090
had free 24 GiB devices, V100 had free 32 GiB devices, and H100/H200 GPUs were
fully allocated. RTX 3090 was selected without a node pin because it was the
smallest-memory adequate available class, not because of its name.

The planned active work was about eight minutes; the 20-minute request added a
12-minute buffer instead of reusing the old fixed ten-minute limit. The exact
route, owner, `salloc`, `srun --pty`, driver argv, modes, and helper hashes are
in `launch-contract.md`. The realized one-shot ledgers are:

```text
salloc_attempts=1
srun_attempts=1
driver_invocations=1
terminal_events_fired=1
terminal_events_handled=1
```

One tmux-owned SSH ControlMaster carried every remote operation and the one
allocation/run owner. One tmux terminal-event source was armed before owner
creation, fired once, and was handled once. The local event SHA-256 is
`ab318c16533e2b532ec5de8aa495ad65fc42737497260f91efafc6c8739826b2`.
The route and its task tmux session retired at `2026-09-01T22:11:36Z`.

No `sbatch`, retry, requeue, cancellation, second owner, duplicate driver,
manual scheduler polling, push, PR, or merge occurred. Protected job `1579631`,
and the active CNVQ owner/job were never queried by a Slurm command. No prior
job, stage, worktree, report, or branch was changed.

## Fail-closed audit corrections

Scheduler mutation remained blocked until every final check passed. Preserved
pre-scheduler audit logs show: an initial remote bundle verification invoked
outside a Git repository; two malformed static-authorization snippets; and
three launch-audit count/`grep` harness errors. Each correction reused the same
hash-bound source, stage, clearance, and helper bytes; the stage and zero
ledgers did not change. The final stage, authorization, helper, mode, output,
environment, and launch audits passed before the sole owner began.

Two post-run result-audit drafts also failed without running GPU code: one had
stray patch markers, and one incorrectly equated five outer trials with five
raw `do_bench(..., return_mode="all")` samples. Both failures are preserved;
the corrected independent audit follows the accepted profiler contract and
passes.

## Artifacts and reproduction

The durable output directory is:

```text
/data/scratch-fast/kwen1/structured-hadamard/rot-phasea-libdevice-fix-gpu-r2-corrected-owner-run/attempt-20260901T214700Z-48a972220979/outputs/phase-a-48a972220979
```

Key hashes:

```text
phase-a.jsonl                 00ce50472f3690ddaffdf79cac06ab78a53b7be52d3e7f4c445f5c130084236a
phase-a.raw-samples.jsonl     a4950eb79de4ac52718af0e304ee18349d41204f796f53d2418bb2845693f5c0
phase-a.execution.json        083a216994cee6695596f255de3d22997fe3e65b0483b116ee23edd3c067fc56
result audit                  e19516943c61886d31418fbd3de7bba71d2ed18e14e4dc149a2bb7dfa7e69e33
GPU execution log             8d0bd0c9d33600bcf1ef6618fe9643441c92700398da641a539180dfac067e20
tmux owner log                8f31394170b07b594abe456cfdd50fc8ec0f1099f0b55579aa25b8a0b05c742b
terminal accounting           778e329615b9944b812e36ac1074a363204168c7fb00282e94f62432d73e3d6f
47-entry evidence manifest    f355e913a6d2230ffd10a88afee09d9794e1d2a96f0af4184e134a7e1f648020
```

The exact committed local copies are under `artifacts/`. Reproduce the
scientific invocation only from the recorded stage, environment, clearance,
and exact owner contract in `launch-contract.md`; this completed task grants no
additional allocation or retry authority.
