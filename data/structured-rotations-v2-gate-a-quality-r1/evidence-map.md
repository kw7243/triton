# Structured Rotations v2 Gate A evidence map

Frozen 2026-09-10 before Experiment A1 outcomes.

| Historical evidence | Transfers to v2 Gate A | Does not transfer |
|---|---|---|
| Phase A, Llama-3-8B, packed W4A4: full rotation improved PPL from 1075.36 to 101.04 and raised decode latency 29.18%. | Full mixing had a large causal quality effect at the FFN down-projection input, and online rotation cost was material on the measured RTX 3090 stack. This justifies re-establishing a matched quality/cost frontier. | The absolute W4A4 result was still unusable versus FP16 PPL 6.82. It lacked PeRQ, used a per-row signed quantizer without matched clipping/error correction, and used a different model/runtime contract. Its PPL values are not a v2 baseline. |
| Phase B: `H32`, `H128`, and full transforms had distinct local errors and costs; calibration-half rankings were stable (rho 0.975). | Block width can create a real quality/cost gap, so both fixed block widths remain informative controls. The old layer cache and rankings show that real 8B activations are heterogeneous. | The local NMSE proxy had only weak six-site PPL correlation (rho 0.257), the calibration was 8K rather than the v2 32K/16K split, and no PeRQ control was present. The per-layer ranking is not reused for v2 layer or method selection. |
| Phase C: the frozen per-layer selector never dominated the fixed-transform frontier and ended at C4. | It closes the tested per-layer transform-allocation hypothesis and prevents spending Gate A effort on another selector. | It says nothing about whether a small static set of representative channels covers the residual error after PeRQ. No selector code, policy, or Phase C outcome is used to construct A1. |
| Preserved branch tips and artifacts. | A/B/C tips and trees match the handoff; all three local artifact manifests rehash successfully; the old remote stages retain their exact scientific source commits. | They remain immutable historical evidence. No old stage is extended, reused as a v2 stage, or treated as a source of v2 outcomes. |

Preserved tips checked read-only:

- A `3bc279eca8728b07f64105a39dcda3e5f896357e`, tree `8b34fdb5dfd9ff241b654ff88dd51951d762f78c`.
- B `f7cd759307ffdb534581587520b42e95a309498c`, tree `43e1d8b804e6c54bc4f93d6b550f0cc5244c3228`.
- C `8c8853f29aefb78a7dfc7c67754343c0ab47f40b`, tree `387558e8f0875b1288fe53273ae7e9c329fb4046`.

The v2 question is narrower and new: after MassDiff PeRQ, is there a material six-layer output-error gap to full rotation, and do one or two fixed representative lanes per block cover the remaining large-activation events? Experiment A1 answers only that quality-side question.

