# Phase A real-model baseline preflight

Status: **blocked before staging and before scheduler mutation**.

The required real-model baseline was not launched. Scheduler attempts are
exactly zero: no scheduler query, persistent SSH route, tmux owner, `salloc`,
`srun`, `sbatch`, scientific driver, terminal event, retry, requeue, or
cancellation occurred. No immutable stage was created because the executable
W4A4 preflight did not clear.

## What passed

The named local branch `fm/rot-phasea-real-model-baseline-r1` began clean at
required commit `0c4aaf075d929be2474fd271ffb8bc1244206d99`, tree
`8728dfa6d4fd17027e0660c1156b6aa47700094e`. Git's strict ancestry check
passed before new work. The authoritative plan SHA-256 is
`4c0f16b28a8c92aa2a70e163df36d2e5a6e98bdcd9c4aac77a0969491f307d45`.

The first-priority candidate was the complete authorized local cache of
`NousResearch/Meta-Llama-3-8B`, snapshot
`315b20096dc791d381d514deb5f8bd9c8d6d3061`. It is a 32-layer BF16 Llama
model with hidden size 4,096, intermediate size 14,336, 32 attention heads,
and 8 KV heads. Its four weight blobs total 16,060,556,376 bytes and resolve
to cache blob SHA-256 identities recorded in `artifacts/preflight.json`.
WikiText-2 raw-v1 test data was also complete at cache revision
`b08601e04326c79dfdd32d625aee71d232d685c3`.

The bounded repository/environment search rejected every surrogate path:

- `torch._int_mm` stores int8 and is not accepted just because values are in
  the signed four-bit range.
- The Triton MLIR example unpacks nibbles to fp16 before `tt.dot`.
- The installed TorchAO paths are A16W4, DA8W4, float8-activation/W4, or
  weight-only int4, not W4A4.
- Fake quantization can provide quality evidence but not accepted latency or
  end-to-end evidence.

Pinned QuaRot commit `5008669b08c1f11f9b64d52d16fddd47ca754c5a` does contain a
scientifically suitable kernel in source: activations and weights are signed
int4 values packed two per byte, and CUTLASS executes `int4b_t × int4b_t`
tensor-core GEMM with int32 accumulation. A deterministic CPU audit at seed
20260901 passed activation and weight pack/unpack, matched an independent
int32 matrix multiplication exactly, and matched scale dequantization exactly.
This proves the arithmetic and packing convention; it does not substitute for
an executable GPU kernel.

## Exact blocker

No importable packed W4A4 runtime exists in the verified environments. The
QuaRot extension and fast Hadamard extension were absent. A separate scratch
overlay and compiler prefix were prepared without modifying the authorized
base environment: Python 3.10.20, Torch 2.8.0+cu128, Triton 3.4.0, CUDA NVCC
12.8.93, CUDA runtime 12.8.90, GCC/G++ 13.4.0, CMake 4.1.0, and Ninja 1.13.0.

The pinned source build was not operational in bounded preflight:

1. CMake on the scratch clone entered `D/rpc_wait_bit_killable`; the build and
   a subsequent scratch-to-local copy were each retired with all related PIDs
   gone.
2. A fresh temporary clone at the same exact parent and submodule commits
   completed CMake. Its first compile found CUDA headers under the conda
   target directory rather than the prefix include directory. Supplying the
   exact include and library paths resolved that toolchain-layout error.
3. The corrected one-worker build then remained in
   `D/mem_cgroup_handle_over_high` at 1,015,676 KiB RSS for more than six
   minutes without producing the first bindings object. It was retired rather
   than continue an unbounded native build on the login node. The final check
   found no build process, QuaRot package, or fast-Hadamard extension.

The model path also was not ready to run unchanged. QuaRot's actual-weight
checkpoint and e2e scripts enumerate Llama-2 only, while its fake-quant path
alone lists Llama-3. The cached Llama-3 model uses grouped-query attention;
the e2e cache code derives head dimension from KV-head count, yielding 512
instead of the required 128. The stock e2e timing script also constructs an
uninitialized quantized model. Therefore it cannot provide either required
actual-weight W4A4 baseline as-is.

Compiling or debugging the runtime inside the sole authorized allocation would
cross the contract's pre-scheduler proof boundary and risk spending the only
attempt on environment work. Implementing a replacement W4A4 kernel is outside
scope. The task therefore fails closed here.

## Scientific disposition

No FP16/BF16 perplexity, W4A4-no-rotation perplexity, rotated W4A4
perplexity, affected-layer timing, rotation timing, or end-to-end decode row
exists. This is not a valid Phase A baseline.

- A1: not evaluated.
- A2: not evaluated; no stress regime is proposed or run.
- A3: not evaluated and cannot be established by this task.

No Phase B/C work, H32/H128 run, learned rotation, lower-bit stress test,
push, PR, merge, or remote branch was performed. The earlier synthetic RTX
3090 experiment was not rerun.

Machine-readable evidence is in `artifacts/preflight.json`.

## Final bounded minimal-binding recovery

The captain subsequently authorized one final recovery: bind only the
unchanged, already CPU-audited QuaRot `gemm.cu` and `quant.cu` sources, with a
small PyTorch compatibility binding and no KV-cache or FlashInfer code. A
replacement kernel, surrogate arithmetic, another implementation path, stage,
or GPU attempt remained prohibited.

The helper and binding bytes were independently SHA-256 matched between this
repository and a fresh scratch preflight directory. The build reverified exact
QuaRot commit `5008669b08c1f11f9b64d52d16fddd47ca754c5a`, CUTLASS commit
`ffa34e70756b0bc744e1dfcc115b5a991a68f132`, and every consumed QuaRot kernel
and interface file digest before invoking the compiler.

The only compiling attempt used one Ninja job, SM86 only, `-O0` for the host
binding, low CPU and idle I/O priority, a 6 GiB virtual-memory limit, and an
immutable 600-second timeout. Two preceding wrapper/configuration exits did
not invoke a compiler: the login node lacks optional `/usr/bin/time`, and
PyTorch required an explicit `TORCH_CUDA_ARCH_LIST=8.6` when no GPU was visible.

The corrected compile invoked only `w4a4_bindings.cpp` first. Its compiler
process reached 1,048,028 KiB RSS while runnable, then remained in the login
node's `mem_cgroup_handle_over_high`/page wait near 1.05 GiB. The 600-second
limit expired before `w4a4_bindings.o` was produced. No QuaRot CUDA source was
compiled, no shared object was linked, no import was possible, and all related
processes were retired. The exact generated Ninja file is preserved locally.

Therefore compile/import, real-model call-path, GPU pack/scale correctness, and
transform/folding smoke were not proven. Per the bounded-recovery instruction,
this is the terminal blocker. No alternative was searched, no stage was
created, and scheduler attempts remain exactly zero. Detailed evidence is in
`artifacts/minimal-binding-recovery.json` and
`artifacts/minimal-build-v2.ninja`.
