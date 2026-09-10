# Gate A quality decision note

Status: A1 contract frozen; outcomes not yet visible.

## Evidence

Pending the one-shot Experiment A1 result.

## Confounds

- This lane is numerical-only. Native segment and end-to-end costs belong to Experiment A2.
- Qwen3-8B is the pre-outcome permitted substitution because the default Llama checkpoint is gated and inaccessible.
- The comparison isolates the FFN down-projection boundary; it is not whole-model perplexity evidence.

## Decision

Pending A1 and the separately owned A2 evidence. Do not enter Gate B from this note alone.

## Exactly one next experiment

Experiment A2 under the separate execution owner: measure the matched native rotation-quantize-pack-consumer and end-to-end baselines, then combine A1 and A2 at Decision A.

