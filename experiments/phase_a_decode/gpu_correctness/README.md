# Phase A GPU correctness-only runner

This integration runs `benchmark.py --correctness-only` from a full repository
snapshot on one Torralba GPU. It exercises `S=96,192`, fp16 and bf16 when the
device supports bf16, all 24 primary units and every secondary index through the
existing exhaustive case, the existing random case, and both kernel configs.

The correctness checks are unchanged: finite outputs, bitwise J gather equality,
bitwise F/H axis equality, and F/H tolerances against the independent scalar-first
PyTorch Hamilton-product oracle.

`validate_environment.py` fails closed before CUDA correctness unless allocation,
GPU, CUDA/PyTorch, memory, source/stage/result paths, and reproducibility metadata
match the declared one-GPU contract. `run_gpu_correctness.sbatch` writes durable
validation, correctness, run metadata, a concise result README, and a hashed final
manifest. `submit_from_stage.py` verifies the staged repository byte-for-byte for
all tracked files, records the stage proof, and contains the milestone's sole
`sbatch` call.

This path never calls `tune`, `measure`, CUDA graphs, latency trials, or `decide`.
It makes only a PASS, FAIL, or NO RESULT correctness classification and cannot make
a broader Phase A/B timing or Gate B decision.
