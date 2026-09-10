# Gate A quality lane

This directory implements only Experiment A1 from the Structured Rotations v2 plan. The immutable choices are in `contract.json`.

The run compares exactly six transforms at six Qwen3-8B FFN down-projection inputs. It recalibrates and reconstructs W4A4 weights from each original BF16 weight, measures a fixed held-out activation sample, and records MassDiff representative coverage. Packed INT4 values are round-tripped before numerical reconstruction. The result is still `numerical_only`: this lane contains no latency benchmark or native low-bit consumer claim.

Run the dependency-free checks from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s experiments/structured_rotations_v2/gate_a/tests -p 'test_*.py' -v
python3 -m compileall -q experiments/structured_rotations_v2/gate_a
```

An experiment must run only from a complete timestamped repository stage directly under `/data/scratch-fast/kwen1/structured-rotations-v2/staging/`. Submit `run_quality.sbatch` with the dynamically selected account, QoS, and partition plus `--parsable --wait --no-requeue --export=NIL`. Results go to the retained root frozen in the contract.

