# Phase A result validation and descriptive analysis

This directory validates the five artifacts written by the Phase A decoder benchmark and
produces a compact table plus a dependency-free SVG. It never emits or infers a
GO/OPTIMIZE/KILL decision or a weight-VQ pivot.

## Contract

The executable schema is in `schema.py` at version `phase-a-analysis-v1`. Validation is
fail-closed and rejects malformed JSON, duplicate JSON keys or CSV headers/cells, non-finite
numbers, missing files, incomplete matrices, inconsistent derived values, and mismatches
between files.

Required input files:

- `correctness.json`: a passed status and unique correctness records covering both S values,
  both input kinds, every tuning configuration, float16, and bf16 whenever bf16 is reported
  supported. Every F/H record includes nonnegative finite max-absolute and relative-Frobenius
  metrics against J, the float32 oracle, and the quantized oracle.
- `tuning.json`: unique configuration names, selection shape Tkv=4096, scores and selected
  configurations for J/F/H at S=96 and S=192. A selected configuration must have a minimum
  score.
- `trial_timings.json`: exactly the six planned (S,Tkv) cells, one or more nondecreasing
  repetition attempts, five randomized-order outer trials per attempt, positive monotonic
  p20/p50/p80 values for J/F/H cold and hot measurements, median aggregates, and J/H hot
  stability values. Aggregates and stability are recomputed.
- `timings.csv`: exactly one row for every product of S={96,192} and
  Tkv={4096,16384,32768}; shape, protocol, configurations, quantiles, throughput, stability,
  correctness summaries, and J/H ratios are type-checked and recomputed.
- `run_metadata.json`: passed status, full git commit and branch, absolute working directory,
  software/runtime/device fields, compute capability, and the declared quaternion and joint-id
  conventions. Synthetic fixtures additionally require a seed and this schema version.

Cross-file validation requires tuning selections, final trial aggregates/repetition, timing
rows, and correctness configuration coverage to agree.

## Outputs

The aggregator reports J/H and F/H ratios for hot and cold p50 latency, J/F/H hot timing
spread, and a descriptive stability label. It writes:

- `analysis_table.csv`
- `analysis_table.md`
- `comparison.svg`
- `validation_report.json`

Synthetic table rows, Markdown, and SVGs are visibly labeled
`SYNTHETIC — NOT A SCIENTIFIC RESULT`. Real inputs receive the label
`DESCRIPTIVE ANALYSIS — NO GATE DECISION`.

## Commands

From the repository root:

```bash
python3 -m unittest discover -s experiments/phase_a_decode/analysis/tests -v
python3 -m experiments.phase_a_decode.analysis.make_synthetic_fixture \
  --output results/2026-08-24-phase-a-analysis-synthetic/inputs
python3 -m experiments.phase_a_decode.analysis.analyze \
  --input results/2026-08-24-phase-a-analysis-synthetic/inputs \
  --output results/2026-08-24-phase-a-analysis-synthetic
```

Generation refuses to overwrite existing fixture evidence or analysis artifacts by default.
