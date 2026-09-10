# Structured Rotations v2 Gate A execution evidence map

| Evidence | What it establishes for A2 | What it does not establish |
|---|---|---|
| Historical Phase A | A real packed W4A4 consumer ran on RTX 3090; full online mixing had material layer and decode cost. | Its signed quantizer, Llama model, poor absolute quality, and unfused timings are not the v2 frontier. |
| Historical Phase B | Full, local-32, and local-128 costs separated at the real FFN boundary. | Its isolated transform/layer timings and weak PPL proxy do not complete A2. |
| Historical Phase C | The old per-layer selector failed to dominate fixed choices. | It does not test PeRQ residual quality or representative-channel bridges. The selector remains closed. |
| v2 A1 job 1818754 | Full beats PeRQ-32 and PeRQ-128 in matched six-layer output error; PeRQ-128 is a strong mandatory competitor. | Numerical reconstruction is not a native consumer or latency result. |
| Preserved QuaRot extension | Physical uint8-packed signed INT4 activation/weight operands are consumed by CUTLASS tensor-core GEMM with int32 accumulation on SM86. | The extension's original symmetric quantizer does not by itself match A1 asymmetric activations; A2 applies and verifies the exact zero-point mapping. |

A2 freezes the three Section 5 row counts at 2048, 1, and 8. It measures the
complete producer-to-consumer segment and end-to-end model latency. PeRQ is
segment-only because only the A1 sampled-layer permutations are frozen.
