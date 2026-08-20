# Failed staging record: Slurm job 1516880

- State: `FAILED`, exit `126:0`, elapsed `00:00:07`.
- Node: `torralba-3090-1`.
- Submitted commit: `b3eb35bd959fc6568170fc2e13ea1e725097673c`.
- Exact failing command: `/bin/bash "$STAGING_HELPER" --staging-parent "$STAGING_PARENT" -- <benchmark command>`.
- Saved stderr: `/bin/bash: /afs/csail.mit.edu/u/k/kwen1/.codex/skills/research-reproducibility/scripts/stage_and_run.sh: Permission denied`.
- The AFS helper was not readable without the kwen1 AFS token, so no staging metadata or benchmark artifact was created.
- `staged_snapshot.txt` is one newline. No experiment ran.
- Raw job artifacts are byte-preserved; checksums are in `artifact_hashes.sha256`.
- Filesystem, ACL, tokenless reproduction, and the noncompliant scratch-mirror control are in `afs_access_evidence.txt`.
