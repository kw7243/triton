# SYNTHETIC — NOT A SCIENTIFIC RESULT

This directory is an isolated deterministic fixture for the Phase A analysis lane. No GPU,
Slurm job, benchmark, or scientific experiment produced these numbers.

- Fixture seed: `20260824`
- Schema: `phase-a-analysis-v1`
- Fixture inputs: `inputs/`
- Generated table: `analysis_table.csv` and `analysis_table.md`
- Generated plot: `comparison.svg`
- Validation and input hashes: `validation_report.json`
- Scientific decision: none

The arbitrary timing values exercise J/H and F/H aggregation only. They must not be copied
into `results/2026-08-20-hurwitz-decode-baseline/` or used to infer a launch gate or a
weight-VQ pivot.
