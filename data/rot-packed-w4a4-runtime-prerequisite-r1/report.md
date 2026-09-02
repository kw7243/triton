# Packed W4A4 runtime prerequisite report

## Status

Complete, local only. One non-GPU `tig-cpu` allocation built and validated the
unchanged QuaRot signed-W4A4 extension, then exited 0. The reusable binary is:

```text
/data/scratch-fast/kwen1/structured-hadamard/rot-packed-w4a4-runtime-prerequisite-r1/attempt-20260902T000347Z-701e372572e8/output/accepted-build/phase_a_w4a4_cuda.so
SHA-256 10a961e8855d7349708412aa8c21a38667fb75c94c372180dd6f992913e0a8e0
size 1,517,328 bytes; regular file; uid 28131; mode 0755
```

No GPU was requested or used. No model inference, scientific driver, timing
row, PPL row, push, PR, or merge was performed. This prerequisite does not
authorize the held Phase A GPU run.

## Source and compatibility preparation

The named local branch
`fm/rot-packed-w4a4-runtime-prerequisite-r1` started directly from required
tip `9590d879af9a0d2e083c53768152fe4fcc7d58a2`. Strict ancestry was verified.
Its immutable source-stage commit is
`701e372572e8e60524b362a7426bc71bf0904bed`; the implementation commit beneath
it is `2e1e5f10b83593f404bdd4e387b9bf95de1baee4`. No earlier branch or worktree
was moved or rewritten.

The extension still binds only the existing minimal PyTorch helper and these
unchanged QuaRot sources:

- QuaRot `5008669b08c1f11f9b64d52d16fddd47ca754c5a`
- CUTLASS `ffa34e70756b0bc744e1dfcc115b5a991a68f132`
- binding SHA-256
  `3b1020fc530ccc934f3cb04d10410a95dda99d2c1b8fc5643fc34979ec7a763f`
- `quarot/kernels/gemm.cu`, `quant.cu`, and their four consumed headers,
  with exact hashes in `artifacts/build-result.json`

The arithmetic boundary is unchanged: signed int4 activations and weights are
packed two per byte, CUTLASS performs int4-by-int4 tensor-core GEMM with int32
accumulation, and dequantization uses explicit row and column scales. No
replacement kernel, int8-container path, fake quantization, weight-only
arithmetic, KV-cache binding, or FlashInfer binding was introduced.

The actual-weight preparation now:

- resolves the cached `NousResearch/Meta-Llama-3-8B` snapshot at revision
  `315b20096dc791d381d514deb5f8bd9c8d6d3061` and uses
  `LlamaForCausalLM.from_pretrained` rather than an uninitialized timing model;
- verifies the index, all 291 expected state keys, and all four weight-shard
  hashes before accepting a load, and rejects meta tensors;
- computes the GQA head dimension as `4096 / 32 = 128`, with cache shape
  `[batch, 8, sequence, 128]`;
- keeps standard eager Transformers attention; and
- installs online full-Hadamard plus packed-W4A4 only at selected FFN
  `mlp.down_proj` modules. The actual down-projection weight is folded with the
  matching full-Hadamard transform before replacement.

The 14,336-wide transform uses the unchanged QuaRot 28-by-28 outer matrix and
order-512 inner Hadamard. The static audit found it orthogonal; its row-major
int8 outer-matrix SHA-256 is
`a6cb994c78f9acd4ce172a154b97e3982594cc0124ce7af90cb88810b9509a8c`.
This task verified the cached bytes and loader behavior but intentionally did
not instantiate the model or run inference.

## Immutable stages and clearance

The complete final local project stage is
`/tmp/rot-packed-w4a4-project-stage-701e372572e8`. A complete scratch copy is
under the attempt root shown below. Both have ordinary independent Git
metadata, exact HEAD `701e372572e8e60524b362a7426bc71bf0904bed`, and read-only
stage contents. The project manifest covers 1,801 entries:

```text
project manifest  a5d99602d4016dbe9bd435d6f7996c228b80c97e4baebcc9901667432a2dbe3e
project metadata  80585c40e6d12e7664eaa7f3a644177033009060bc7d13e32b77adc422e529db
```

The separate dependency stage is
`stages/dependency/QuaRot` beneath the attempt root. QuaRot and CUTLASS each
have ordinary independent Git metadata at the exact commits above, clean
working trees, complete commit connectivity, and no source changes. Its full
5,728-entry byte manifest has SHA-256
`8f6037a6385f874253c69bc3ed5bf843480783984def548941bfdf1018c354ec`.

The immutable attempt root is:

```text
/data/scratch-fast/kwen1/structured-hadamard/rot-packed-w4a4-runtime-prerequisite-r1/attempt-20260902T000347Z-701e372572e8
```

The fail-closed clearance binds the project and dependency stages; exact
model bytes; owner; output; helper modes and hashes; compiler and inspection
tools; Python packages; environment; build argv; scheduler argv; and terminal
ledger. Its SHA-256 is
`52b6adc17981e999812cde0b96ac36ab79904a75c0b3ff0c79a0fbd256f69aac`.
An independent prelaunch audit rehashed 1,801 project files, 5,728 dependency
files, six helpers, eight tools, and every model file. It passed with SHA-256
`42393932cd395abd89c3e9b7bb16b436d2c8c207f64eaaafbff6c9c40399ad8e`.

The reused isolated runtime was verified as Python 3.10.20, Torch
2.8.0+cu128 (CUDA build 12.8), Transformers 5.12.1, Triton 3.4.0, Ninja
1.13.0, and CMake 4.1.0. The Python executable SHA-256 is
`fa10ee8f4c18e62cbd1e467c156a228be45138bd537b9949e66fc8e5937a018e`.
The compiler prefix contains NVCC 12.8.93 and GCC/G++ 13.4.0, with executable
SHA-256 values `5ad3b681e5f65dc6e66799ba019627986b618120778a9cccc33276ad6893a673`
and `f3043ba4d6dce732378de6c4c6ebb62ff89229e90dc982df6ed1071396e66029`.
The complete environment identity has SHA-256
`2b8775c36f1aa88ef3de69b1a644a0351c308581e12373a883f23d9c8369e89a`.
No system or authorized base environment was changed. A first isolated
`cuobjdump` package solve failed before scheduler use because its channel set
lacked `libgcc-ng`; the preserved fresh `v2` prefix succeeded and supplied
cuobjdump 12.8.90. Both outcomes are recorded in
`artifacts/cuobjdump-prefix-preparation.json`.

## Sole CPU owner

Read-only association discovery selected the smallest adequate established
non-GPU association: account `csail`, QoS `tig-main`, partition `tig-cpu`.
The preserved login-node evidence suggested a 15–25 minute, 4–6 GiB build;
the sole request used two CPUs, 16 GiB, 45 minutes, with one Ninja worker.
The exact scheduler argv was:

```text
/usr/bin/salloc --account=csail --qos=tig-main --partition=tig-cpu --job-name=rot-w4a4-cpu-build --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=16G --time=00:45:00
/usr/bin/srun --pty --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=16G --kill-on-bad-exit=1
```

Allocation `1662121` ran on `groenig-2.novalocal` in `tig-cpu`, with two CPUs,
16 GiB, and no `CUDA_VISIBLE_DEVICES`. It began at
`2026-09-02T00:21:22.567858Z` and reached its one terminal event at
`2026-09-02T00:24:31.021559Z`; the owner exited 0 and Slurm relinquished the
allocation normally. The exact ledger is one owner attempt, one `salloc`, one
`srun`, one payload, and one terminal event. There was no `sbatch`, retry,
requeue, cancel, second owner, second allocation, GPU request, `nvidia-smi`,
or protected-job inspection.

During the active allocation, one read-only SSH build-log observation was made
and is preserved in `artifacts/supervision-observation.json`. After the
secondmate supervision correction, supervision consumed only the existing
owner terminal stream/event: there were no further ad hoc reads or scheduler
queries. The authority attribution is recorded in
`artifacts/authority-correction.json`. The single persistent local-tmux SSH
route, owner, allocation, terminal event, and socket were all retired.

## Artifact acceptance

The clean build directory was new for this attempt. Build timing from Ninja
was about 96.2 seconds for the binding, 29.4 seconds for GEMM, 9.3 seconds for
quantization, and 1.8 seconds for the link. `torch.utils.cpp_extension`
generated Ninja directly; CMake was not used. `build.ninja` has SHA-256
`eee8fb5f02cc7f2fd64f6ae6fdb78d137134b1e0afd8084fb2d9b0ffbabdccef`.
All 958 compiler dependency paths are covered by the immutable manifests.
Object creation times postdate allocation start, every object is hashed in
`artifacts/build-result.json`, no failed login-build path occurs in Ninja or
its dependencies, and the contamination audit is false.

Inspection found resolved linked libraries and the Python initialization
symbol. Import succeeded with no CUDA device operation and exposed
`sym_quant`, `matmul`, and `sym_dequant`. The postbuild audit passed with
SHA-256 `247074cc4b350679090e8b28e7459b985fc2ffac3153bf6a27b9ad9c613e145c`.
The final 35-entry scratch evidence manifest has SHA-256
`142a67455e39fc1ca6333da50fef569524901892a262988bab5684f4905f3bcf`.

Architecture inspection found exactly:

- native SASS: `sm_80`, `sm_86`;
- forward-compatible PTX: `compute_80` (`.target sm_80`);
- no native SM89 or SM90 claim; and
- no SM70/V100 support.

A later dynamically selected accelerator must either be SM80/SM86 or pass a
separate compute_80 PTX JIT validation before this artifact is used. This task
does not claim that validation for any newer accelerator.

## Tests and reproduction

The bounded CPU/static suite has 54 passing tests. It covers configuration,
GQA/cache shapes, exact weight loading, module replacement and call routing,
standard-attention preservation, transform/inverse/folding references,
packing/scales/dequantization, architecture boundaries, and CLI/config
assembly. It passed both locally and inside the sole allocation. Per the
project instructions, no `make` was run because these are Python-only source
changes plus a separately built external extension.

To reproduce, verify the hashes in `artifacts/clearance.json`, materialize the
two immutable stages at their recorded commits, and use the exact Python,
toolchain, environment, and build argv recorded there. The build argv is the
verified Python followed by
`build_w4a4_extension.py --quarot-root <dependency-stage> --build-directory
<new-output> --cuda-home <cuda-build-tools-12.8.93> --cxx
<cuda-build-tools-12.8.93/bin/x86_64-conda-linux-gnu-g++>`, with
`MAX_JOBS=1` and `TORCH_CUDA_ARCH_LIST=8.0+PTX;8.6`. Use the exact non-GPU
scheduler argv above, then run the recorded unittest discovery command and
repeat the link, symbol, import, architecture, dependency-coverage, and
contamination audits. Large outputs and the full dependency/Ninja manifests
remain under the immutable scratch attempt; small handoff evidence is in
`artifacts/` beside this report.

The result-evidence commit is
`bb642cf908cb891da385ef1750cc69878e4f483e`, tree
`d4355892e63f20983a4f392fc0a93da84d20b343`. The final local branch tip is the
following report/notebook provenance-pointer commit and is reported at
handoff.
