# Structured Rotations v2 research notebook

## 2026-09-10 — Gate A quality intake and contract freeze

- Branch: `fm/structured-rotations-v2-gate-a-quality-r1` from exact base `f893845b9b91599ebd3b7a9c7f28164f39c7ed94`. No rebase, merge, push, or PR is permitted.
- Forward authority: `idea7_structured_rotations_experimental_plan_v2.md`, Gate A / Experiment A1 only.
- Historical A/B/C evidence was checked read-only. The clean tips and trees match the direct-handoff report, and all 22 A, 29 B, and 21 C local artifact-manifest entries rehashed successfully. Remote A/B/C scientific stages remain under their original scratch paths; their recorded source identities were rechecked without mutation.
- The old Phase C result closes only the rejected per-layer selector. No Phase C code or policy is imported into this branch.
- Default `meta-llama/Llama-3.1-8B` is gated and no Hugging Face credential is installed on CSAIL. Before any outcome, the one permitted substitution was frozen: complete cached `Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218`.
- Exact Experiment A1 choices, model/data hashes, quantization and clipping parity, six layers, activation reservoirs, metrics, paths, budget, and exclusions are frozen in `experiments/structured_rotations_v2/gate_a/contract.json`.
- Calibration uses exactly 32K tokens from WikiText-2 train. Development uses exactly 16K tokens from the disjoint validation split. Final test data is untouched.
- The six fixed controls are identity, full exact Hadamard, local `b=32`, local `b=128`, MassDiff PeRQ plus local `b=32`, and MassDiff PeRQ plus local `b=128`.
- Execution kind is `numerical_only`: every W4A4 tensor is packed/unpacked exactly for reconstruction, but no native timing or low-bit consumer claim belongs to this lane.
- Staging must be a fresh complete repository copy under `/data/scratch-fast/kwen1/structured-rotations-v2/staging/`. Results and retained activation/statistics caches must be under `/data/vision/torralba/u/kwen1/structured-rotations-v2/`.
- Exactly one next experiment is frozen: A2 native timing under the separate execution owner, followed by the combined Decision A. This lane will not run it.

