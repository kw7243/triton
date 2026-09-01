# Phase A compatible-GPU kernel microprofile

Status: **failed: the sole hardware attempt reached one correctly visible RTX
3090, then the untimed A4 Triton kernel failed to compile against installed
Triton 3.4.0.** The accepted execution driver was not invoked. No timing rows
or important-kernel gate result exist.

## Scope and provenance

- Task: `rot-phasea-kernel-profile-r1`
- Local-only branch: `fm/structured-hadamard-phase-a-kernel-profile-r1`
- Pre-run commit: `11c6d87772de27d97a7d5b1f1f9577d70b4b41ca`
- Exact required parent: `21ea761c61a5e3062cea28cabda43ac04bd5278b`
- Accepted Phase A ancestor: `d57acb60db2a4507bbff984fb3c9771e8a6ada3d`
- Stage manifest digest:
  `ec8e4c6f926bef98441b13678bcd6024b4ded27287f8ba96865889215c44045b`
- Stage:
  `/data/scratch-fast/kwen1/structured-hadamard/rot-phasea-kernel-profile-r1/attempt-20260901T164930Z-11c6d87772de/stage/20260901T164930Z-11c6d87772de-code`

The fresh stage has an ordinary independent `.git`, full object connectivity,
the exact commit/tree, 1,756 committed manifest entries, and only its two
declared reproducibility control files untracked. The owner-only mode-0600
clearance bound that stage, commit, manifest, `cuda:0`, the new output path,
the fixed synthetic `[1,11008]` seed-0 workload, immutable metadata pins, and
the accepted timing configuration. No model or dataset bytes were downloaded.

The accepted matrix remained exactly `I` and `Hfull`, each with
`transform-only` and sequential `transform+quantize`. Every row would have
used `fusion="none"` and `scientific_evidence=false`; `H32` and `H128` remained
CPU/reference-only.

## Dynamic selection and actual allocation

The read-only snapshot found six free 24 GiB RTX 3090 devices on
`torralba-3090-2`. `vision-torralba-rtx3090` was the smallest-memory suitable
Torralba partition allowed by both the required account and interactive QoS.
V100, H100, and H200 devices had more memory; shared smaller-device partitions
did not allow the required QoS. The request selected only the partition and
did not pin a node.

- Job: `1659613`
- Node: `torralba-3090-2`
- Partition/account/QoS:
  `vision-torralba-rtx3090` / `vision-torralba-urops-meng` /
  `vision-torralba-interactive`
- Resources: one node, one task, one GPU, 2 CPUs, 8 GiB RAM, 10 minutes
- Actual GPU: NVIDIA GeForce RTX 3090, Ampere/SM86
- UUID: `GPU-82e6108f-1eeb-47f3-d820-ad67a7a6bb15`
- VRAM: 24,576 MiB
- NVIDIA driver: `580.178.04`
- `CUDA_VISIBLE_DEVICES`: set non-empty to `0`
- Runtime stack: Torch `2.8.0+cu128`, Torch CUDA `12.8`, Triton `3.4.0`

The Python preflight reached the correctness call only after requiring exactly
one Torch CUDA device and completing and synchronizing a real CUDA tensor
operation. Its final JSON record was intentionally emitted only on complete
success, so the observed compute-capability tuple was not serialized before
the later compiler failure. The actual allocated model is an RTX 3090
(SM86), but this run does not contain a completed driver hardware record.

## Failure

The first quantizer call in the one untimed Triton correctness invocation
failed during JIT dependency discovery:

```text
AttributeError: module 'triton.language.extra' has no attribute 'libdevice'
```

The staged A4 kernel calls `tl.extra.libdevice.rint`. In installed Triton
3.4.0, `triton.language.extra.__init__` exposes backend packages, and the CUDA
implementation containing `rint` is under `tl.extra.cuda.libdevice`. This is a
driver/runtime API compatibility defect, not a visibility or capacity failure.
It occurred on the identity-plus-quantize correctness path before `Hfull`
correctness, before timed rows, and before the accepted driver CLI invocation.

The authorized attempt was consumed exactly once:

- `salloc_attempts=1`
- `srun_attempts=1`
- no `sbatch`, retry, cancellation, requeue, second owner, or Slurm polling
- the single terminal marker fired once, was handled once, and was retired

## Timing and important-kernel gate

The fixed timing configuration was 25 ms warmup, 200 ms repetition, five outer
trials, device-event synchronization, and p10/median/p90 summaries. No timed
driver invocation occurred, so all four accepted rows have no samples or
summary:

| Timing identity | Transform | Samples | Summary |
|---|---|---:|---|
| transform-only | I | not produced | not produced |
| transform-only | Hfull | not produced | not produced |
| transform+quantize | I | not produced | not produced |
| transform+quantize | Hfull | not produced | not produced |

The same-device important-kernel overhead
`(Hfull transform+quantize / I transform+quantize - 1) * 100%` cannot be
computed. The plan's `>=5%` gate is **not evaluated**, neither pass nor fail.
The identity transform-only no-op is not used as a latency denominator.

No RTX-3090-specific latency or overhead was established despite allocating
an RTX 3090, and nothing here generalizes a hardware cost gate across GPU
architectures. A later separately authorized task would first need a
version-compatible A4 rounding call and a GPU correctness gate, then a new
target-specific attempt.

## Evidence boundary

This attempt produced failure diagnostics only, not synthetic kernel
measurements and not model/PPL/end-to-end evidence. The output directory is
absent. No 8B model, weights, data, PPL, end-to-end decode, Phase B/C, or
H32/H128 GPU work ran.

The r3 conclusion is preserved exactly: allocation `1638476` reached
`torralba-v100-1` and exposed one V100 through successful `nvidia-smi`, while
its strict payload failed because `CUDA_VISIBLE_DEVICES` was empty and
assigned-device count was zero.

All durable evidence is under:

```text
/data/scratch-fast/kwen1/structured-hadamard/rot-phasea-kernel-profile-r1/attempt-20260901T164930Z-11c6d87772de
```

Key SHA-256 values:

- Evidence manifest: `db23c4288c03b5420d04800f6aadcd01e620e66d1df01bc1caf7a41b3aed1594`
- GPU execution log: `1c76caa84094792e5622999ed5629d84e914700f6e3f3d74b4cfa86aed3e4284`
- Full tmux pane log: `0aa39af6506f1873d52a208a62fcb7a2a67dba674cccd6e2bf3dad0c8905cb86`
- Terminal event: `fb209cb7b997a71da58e05a1d0ea8a4e3fe405cd1b9fd3c4e8473200dad6c131`
- Task status: `7f38589c660ff9c3494f1a32a9b19f9ef9c45c62f57d88c1cbac2db578095312`
- Scheduler selection snapshot: `458cb15065dc91b222704df7f0c3b7a506acaa2efc49127e3ac766e4f2dee6cd`
- Clearance: `48fb5cb360d57de604ea725318a43f878f2a797013a97b34b39f6af33c72318c`
