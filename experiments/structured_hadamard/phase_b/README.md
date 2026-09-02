# Phase B per-layer quality–latency map

This package executes only the plan's 32 Llama-3 FFN `down_proj` input sites
for `I`, `H32`, `H128`, and `Hfull`.

Frozen scientific semantics:

- Calibration is exactly 16 sequences by 512 tokens, split into deterministic
  disjoint 8-sequence halves. It never expands in this lane.
- `H32` and `H128` are normalized Sylvester FWHTs over consecutive blocks with
  the same symmetric transform folded into each down-projection weight row.
- `Hfull` reuses the accepted pinned QuaRot factorization from Phase A.
- The transform and accepted packed signed-W4A4 layer execute sequentially with
  `fusion="none"`.
- The primary sensitivity score is `NMSE(H32) - NMSE(Hfull)`. Ranking stability
  is Spearman correlation across the two calibration halves, with `rho >= 0.5`
  predeclared as stable.
- The six frozen PPL sites are the three highest and three lowest primary
  scores. Every validation replaces `Hfull` with `H32` at one site only.
- If the six-site PPL check rejects the primary proxy, the driver switches once
  to the predeclared short-sequence end-to-end NLL proxy. No other proxy exists
  in this lane.

The GPU entry point is `driver.py`. `gpu_owner.py` owns exactly one `salloc`
and one `srun --pty`; `local_event_owner.py` owns the sole tmux/SSH terminal
event. The inherited Phase A standard-library preflight audits the new project
stage, accepted Phase A stage, pinned QuaRot/CUTLASS dependency, exact caches,
accepted extension, environment bytes, argv, output parent, and zero ledger.

Outputs include 128 machine rows, raw timing samples, a cache manifest, frozen
selection, six-site validation, ranking/stability analysis, CSV table, SVG
figure, result summary, and a literal B1/B2/B3/B4 conclusion. This package does
not implement or enter Phase C.
