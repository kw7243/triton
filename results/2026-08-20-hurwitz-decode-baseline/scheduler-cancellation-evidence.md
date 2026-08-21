# Scheduler cancellation evidence

Observed from `slurm-login-0.csail.mit.edu` on 2026-08-21. No queue state was changed.

## Slurm accounting

| Job/step | State | Exit | Derived | Reason | Start | End | Node |
|:--|:--|:--|:--|:--|:--|:--|:--|
| `1517414` | `CANCELLED by 0` | `0:0` | `0:0` | `None` | `2026-08-20T08:37:37` | `08:37:39` | `torralba-3090-2` |
| `1517414.batch` | `CANCELLED` | — | — | — | `08:37:37` | `08:37:41` | `torralba-3090-2` |
| `1517414.extern` | `COMPLETED` | `0:0` | — | — | `08:37:37` | `08:37:39` | `torralba-3090-2` |
| `1524492` | `CANCELLED by 0` | `0:0` | `0:0` | `None` | `2026-08-21T00:16:44` | `00:16:50` | `torralba-3090-1` |
| `1524492.batch` | `CANCELLED` | — | — | — | `00:16:44` | `00:16:52` | `torralba-3090-1` |
| `1524492.extern` | `COMPLETED` | `0:0` | — | — | `00:16:44` | `00:16:50` | `torralba-3090-1` |

Neither `slurm-1517414.out` nor `slurm-1524492.out` exists. No Phase A job was active at inspection. Slurm exposes no reason, admin/system comment, or event explaining either cancellation; controller and daemon logs are not readable by this account.

## Staged worktrees

Both worktrees are clean, contain 1,492 tracked files, and have:

- HEAD: `b8d099336ca65cf7aaa1a1b664980f19f5e4e474`
- tree: `89d495c8884a9ec51187238d6da06123908591e1`

| Job | Worktree | Metadata SHA-256 |
|:--|:--|:--|
| `1517414` | `/data/scratch-fast/kwen1/compute-native-vq/staging/20260820_031814-56491e-b8d0993-code/worktree` | `d08143ba231ef622ea744ad2642e474b87ab7ad46655fb0406f00260fc86a876` |
| `1524492` | `/data/scratch-fast/kwen1/compute-native-vq/staging/20260820_193950-fad1f6-b8d0993-code/worktree` | `251c621debadb354c1fd303336eb0ed8bf6a493d621354b943293f34cb0f964b` |

## Durable run-state hashes

```text
b820531f6282749a3febfa1b1d78a6b44dc48cf2d87ab7fc7d43e6410fa2338d  phase-a-stage-5d29df20-r6.sh
13f44bb053bc89fd9396b3a633d562225be9db714163e61ee8b95e54311c8978  phase-a-stage-5d29df20-r6.log
4d00bee263839f4b45ebce493ce9cb28df765e1e38d8c199b6021761977e323f  phase-a-stage-5d29df20-r6.status
62a99877982cac50c62b4cdc7c2986f05752ca7f21b6f5b8881077b1ec82de69  phase-a-stage-5d29df20-r6.helper-output
2cc3dff33f0409dfd70606bb2d923ab5a9e9255f5fb0fbf21288a719c1d3570d  phase-a-stage-5d29df20-r7.sh
23f46f2dceb133e463f861ce66e3b1f522702cbab4652423013a4f79936c6e14  phase-a-stage-5d29df20-r7.log
0ee3d88262446ed1ebfe47d3a422833fb880f60310468619e03d14686f923a1f  phase-a-stage-5d29df20-r7.status
4624a17deb689a01f91b5caca32e24845308656b9aed2e34a870b652bcd9d9aa  phase-a-stage-5d29df20-r7.helper-output
```

The files remain under `/data/scratch-fast/kwen1/compute-native-vq/run-state/`. All earlier partial stages and r1–r5 evidence also remain preserved.

## Blocker

Two independently staged Torralba jobs were forcibly cancelled by Slurm UID 0 before the benchmark emitted output. The available evidence does not identify why.

Smallest next action: a CSAIL Slurm administrator inspects controller/daemon logs for jobs `1517414` and `1524492`, then clears the forced cancellation or identifies an allowed Torralba account/QoS/partition invocation. Do not submit another job until that action occurs.
