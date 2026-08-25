# Phase A submission-free preflight

This lane statically verifies the frozen Phase A launch contract. It is CPU-safe:
it does not import Triton or PyTorch, contact Slurm, stage a repository, run the
benchmark, inspect jobs, or write results. It authorizes no submission.

Run the live repository check from the named scratch worktree:

```bash
python3 experiments/phase_a_decode/preflight/validate_preflight.py
```

Run the labeled fixture suite:

```bash
python3 experiments/phase_a_decode/preflight/validate_preflight.py \
  --fixtures-dir experiments/phase_a_decode/preflight/fixtures
```

The live check verifies:

- the named branch, exact base ancestry, clean frozen source checkout, and
  unchanged benchmark and batch-wrapper hashes;
- the full-copy staging metadata schema, exact source identity, clean source
  status, absolute staged path, and source/stage inequality;
- the Torralba account, QoS, partition, node/task/GPU/CPU/memory/time request;
- the absolute result directory and Slurm log paths plus every expected output;
- strict shell mode, required staging guards, and unmasked final benchmark exit.

The fixture suite includes one valid full-stage observation and labeled negative
cases for missing metadata, a compact-capsule source, equal source/stage paths,
scheduler drift, a relative result path, and fail-open shell behavior. Expected
rejections count as fixture-suite passes.

## Administrative gate

This preflight does **not** show that the repeated `CANCELLED by 0` condition is
fixed. The smallest new scientific evidence needed before considering another
retry is one administrator-authenticated diagnosis tied to job `1570434` that
states the controller/daemon cancellation cause and either confirms the
condition was cleared or supplies the exact permitted Torralba invocation.
Governance still separately requires explicit authorization for a fourth Phase A
submission. Without both, the launch remains closed.

`contract.json` is the concise frozen manifest. The validator pins its critical
values in code as a second fail-closed layer, so editing the manifest cannot
silently relax the contract.
