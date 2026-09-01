# Corrected Phase A launch contract

- Source/driver/transform commit: `48a972220979197359a324ab102eb8de24ce321f`
- Source tree: `ba5f873fb9c2defa7b3af15b971cf5fb8d3092fb`
- Stage: `/data/scratch-fast/kwen1/structured-hadamard/rot-phasea-libdevice-fix-gpu-r2-corrected-owner-run/attempt-20260901T214700Z-48a972220979/stage/20260901T214700Z-48a972220979-code`
- Canonical stage manifest SHA-256: `e54ce37b0f14838731f6458631801e9e2130b66589e6beca27b5a1d615ce903e`
- Manifest-file SHA-256: `d5096b74326048d4ddb84607ffe43552a4372fdd715aa7ff4d1de75346c2a233`
- Stage metadata SHA-256: `d446d626eaea2ba6b03ee0963d19e7bde87c413a8155b62c32394b97b3fc95da`
- Clearance SHA-256: `fec5407404af09cfad8eb3eabb5ff7fb5b62bc20adf92da7d45d7bc7f3c1f402`
- Execution config SHA-256: `d9b0171c40414616dff003cd407677a4d0e71bca38b32f10d8e847a9ca6cc26c`
- Output: `/data/scratch-fast/kwen1/structured-hadamard/rot-phasea-libdevice-fix-gpu-r2-corrected-owner-run/attempt-20260901T214700Z-48a972220979/outputs/phase-a-48a972220979` (must be absent at launch)
- Python: `/data/scratch-fast/kwen1/micromamba/root/envs/causal_forcing/bin/python`; Python 3.10.20; executable SHA-256 `fa10ee8f4c18e62cbd1e467c156a228be45138bd537b9949e66fc8e5937a018e`
- Runtime: Torch `2.8.0+cu128`, Torch CUDA `12.8`, Triton `3.4.0`
- Timing: 25 ms warmup, 200 ms repetition, five outer trials
- Workload: deterministic seed-0 `[1,11008]` contiguous fp16 input; GPU rows only `I` and `Hfull`
- Quantization: dynamic per-row symmetric signed A4 `[-7,7]`, nearest-even; sequential transform-plus-quantize; `fusion="none"`
- Estimated active runtime: about 8 minutes for allocation entry, JIT/correctness, four timing rows, validation, and durable writes. Requested wall time: 20 minutes, including a 12-minute buffer.

## Dynamic selection

Account `vision-torralba-urops-meng` and QoS `vision-torralba-interactive` allow the four dedicated Torralba GPU partitions. The 2026-09-01T21:55:42Z snapshot found free 24 GiB RTX 3090 capacity and free 32 GiB V100 capacity; H100 and H200 GPUs were fully allocated. `vision-torralba-rtx3090` is selected without a node pin because 24 GiB is the smallest adequate available class. No job listing or protected/CNVQ job was inspected.

## One route and one owner

The one tmux-owned ControlMaster route is session `rot-phasea-corrected-owner`, window `ssh-route`, socket `/tmp/rot-phasea-corrected-control.Umqlas/ssh-control`, launched exactly as:

```text
/usr/bin/tmux new-session -d -s rot-phasea-corrected-owner -n ssh-route -- /usr/bin/ssh -M -N -S /tmp/rot-phasea-corrected-control.Umqlas/ssh-control -o ControlMaster=yes -o ControlPersist=no -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o ExitOnForwardFailure=yes slurm-login.csail.mit.edu
```

The single terminal event source waits on tmux channel `rot-phasea-corrected-terminal` before owner creation. The one allocation/run owner will launch exactly as:

```text
/usr/bin/tmux new-window -d -t rot-phasea-corrected-owner -n allocation-owner -- /usr/bin/script -qefc /home/ubuntu/.treehouse/triton-ff92c5/10/triton/data/rot-phasea-libdevice-fix-gpu-r2-corrected-owner-run/local-owner.sh /tmp/rot-phasea-corrected-control.Umqlas/tmux-owner.log
```

Local owner `local-owner.sh` is mode `0700`, SHA-256 `e3d29ef95b5e6ad77166ac778185d49e63a074f7d1d2dfcc13006d3442550382`, and opens one `-tt` channel over the existing ControlMaster to the exact remote owner.

## Exact scheduler and scientific argv

The only `salloc` argv is:

```text
/usr/bin/salloc --account=vision-torralba-urops-meng --qos=vision-torralba-interactive --partition=vision-torralba-rtx3090 --job-name=rot-phasea-corrected-owner --nodes=1 --ntasks=1 --cpus-per-task=2 --gres=gpu:1 --mem=8G --time=00:20:00 /data/scratch-fast/kwen1/structured-hadamard/rot-phasea-libdevice-fix-gpu-r2-corrected-owner-run/attempt-20260901T214700Z-48a972220979/run-state/srun-owner.sh
```

The at-most-one `srun --pty` argv is:

```text
/usr/bin/srun --pty --nodes=1 --ntasks=1 --cpus-per-task=2 --gres=gpu:1 --kill-on-bad-exit=1 /data/scratch-fast/kwen1/structured-hadamard/rot-phasea-libdevice-fix-gpu-r2-corrected-owner-run/attempt-20260901T214700Z-48a972220979/run-state/gpu-payload.sh
```

After the one-device visibility and untimed corrected-A4 gate passes, the only scientific driver argv is:

```text
/data/scratch-fast/kwen1/micromamba/root/envs/causal_forcing/bin/python -m experiments.structured_hadamard.phase_a.execute --scheduler-clearance-file /data/scratch-fast/kwen1/structured-hadamard/rot-phasea-libdevice-fix-gpu-r2-corrected-owner-run/attempt-20260901T214700Z-48a972220979/run-state/clearance.json
```

No `sbatch`, retry, requeue, cancellation, second owner, node pin, or conversational scheduler polling is authorized.

## Remote executable identities

All are owned by UID 28131 and mode `0700`:

```text
99b615b5d44a1ce2369ea1dfd7f86738e66ad5f84038ab6a19c24faece71e9f3  stage_audit.py
8c027b09c2fcdf1fb8235add2f5e838c3f7004b5d2d02b61265723c64cddf20e  ledger.py
b7ee8f1a720d2fb106b0fe52223adbfb0f3fc7244898ef7454f2ab281d5a28b0  gpu_preflight.py
14fa4b978aaeb4dd4d6ab8ca36fd95aed5766d91c645192dae74b7dea97fd221  gpu-payload.sh
b73877224fa3ef8a2525419b6d9ae6d9e3f6f87761f2615aa6ade2c6032e0ac0  srun-owner.sh
3f3834fa82a507fc57dd54ee3516adf0057068ccc6c1776c929594d1a6739935  allocation-owner.sh
7115688d3d46aaf2f2899aec1729dce799eb769ae466e93b24517ac5840bb0ec  remote-attempt.sh
```

Initial one-shot ledgers are exactly zero for `salloc_attempts`, `srun_attempts`, `driver_invocations`, `terminal_events_fired`, and `terminal_events_handled`.
