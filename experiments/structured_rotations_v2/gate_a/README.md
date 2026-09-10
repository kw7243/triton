# Structured Rotations v2 Gate A

`contract.json` is the read-only A1 quality contract. `execution_contract.json`
freezes A2. A2 is valid only from a complete timestamped CSAIL stage containing
the pinned native W4A4 artifact and frozen A1 layer-0 cache under `inputs/`.

The batch entry point is `run_execution.sbatch`. Do not run `execution.py` from
the source worktree or a login node. Before submission, create and verify
`STAGE_FILE_MANIFEST.json`, inspect current Slurm capacity and matching attempts,
and run the exact allocation tuple through `sbatch --test-only`.

A2 times identity, exact full Hadamard, local-32, local-128, and the runnable
PeRQ segment controls with one common native packed consumer. It does not
contain a bridge, selector, quality evaluation, or Gate B path.
