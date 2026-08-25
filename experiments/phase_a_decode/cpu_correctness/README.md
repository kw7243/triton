# Phase A CPU correctness lane

This directory is an independent, deterministic PyTorch oracle for the Phase A
J/F/H Hurwitz decoder. It is deliberately CPU-only and performs no Triton
compilation, CUDA call, benchmark, scheduler action, or performance inference.

The contract is:

- quaternion components are scalar-first `(w, x, y, z)`;
- `id = p*S+s`, with 24 primary Hurwitz units;
- J gathers a precomputed joint table;
- F gathers both factors and evaluates the generic Hamilton product;
- H uses exact signed permutations for the first 8 units and signed half-sums
  for the remaining 16 units.

The tests exhaust every primary unit and every secondary index for representative
`S = 1, 2, 7, 31`, exercise all 16 half-unit sign masks, cover basis,
zero/signed-zero, mixed-sign, small/large power-of-two edge values, validate
fp16 and bf16 table-storage round trips, and run reproducible random properties
with seeds `0`, `20260824`, and `0xC0FFEE`.

Run from the repository root with GPU visibility disabled:

```bash
CUDA_VISIBLE_DEVICES="" python -m unittest discover \
  -s experiments/phase_a_decode/cpu_correctness \
  -p 'test_*.py' -v
```

## Integration seam

`experiments/phase_a_decode/benchmark.py` combines the mathematical definitions
with top-level Triton and plotting dependencies, and its executable decode path
allocates CUDA tensors. This lane therefore does not import that module on a
CPU-only login node. It independently fixes the same boundary objects:

- `primary_units()` order;
- scalar-first `hamilton(left, right)`;
- flat `id` decomposition;
- joint-table layout `(24*S, 4)`;
- F's generic product and H's two specialized cases.

A GPU integration test can use `oracle.decode_variants()` as the seam: construct
the same stored secondary table and flat ids, copy those inputs to the device,
run the benchmark J/F/H kernels, and compare their copied-back outputs to the
CPU results. That adapter belongs with a GPU-only test lane and is intentionally
absent here.

## Numeric expectations

Float32 J/F/H comparisons use `atol=rtol=2e-7`; axis signed permutations are
bitwise equal. Storage round trips use `atol=rtol=2e-3` for fp16 and `2e-2` for
bf16, covering both stored-input and stored-output rounding. These checks are
tolerances for correctness only and make no performance or experiment claim.
