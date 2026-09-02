# Phase C five-budget Pareto selector

This package consumes only the accepted Phase B 32-layer FFN `down_proj` map.
It freezes one deterministic adjacent-downgrade trace from `Hfull` to `I`, then
selects the earliest trace prefix at or below each 0/25/50/75/100% transform-cost
budget. The exact tie key is `(rho, layer, target-transform rank)`. Lower budgets
are longer prefixes of the same trace, so feasibility and assignment monotonicity
are structural rather than post-hoc checks.

`policy-freeze.json` is generated and committed before the scientific driver can
evaluate any policy. It contains every layer assignment, the exact Phase B source
row hash used for that assignment, target and realized predicted cost, fixed
baselines, endpoint aliases, and unique measurement IDs. The driver re-derives
the entire freeze from the accepted Phase B bytes before loading outcomes.

The GPU driver evaluates each unique assignment sequentially with the inherited
packed signed-W4A4 runtime, real `I`/`H32`/`H128`/`Hfull` transforms, and
`fusion="none"`. It releases each model before constructing the next policy.
WikiText-2 PPL and batch-1 decode settings are exactly those accepted in Phase A.

No pinned, already-reproducible comparable MixQuant/PeRQ implementation exists
in the accepted repository/dependencies, so that comparison is recorded as
unavailable and does not block the five-point curve. This lane does not enter
Phase D, C-kernel, accuracy recovery, kernel fusion, or a second scientific run.
