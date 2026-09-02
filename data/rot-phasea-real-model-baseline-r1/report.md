# Phase A real-model baseline — completed

Status: **A1 met on the sole real-model GPU attempt**.

The distinct Phase A gate completed on cached
`NousResearch/Meta-Llama-3-8B` revision
`315b20096dc791d381d514deb5f8bd9c8d6d3061` and WikiText-2 raw-v1 test
revision `b08601e04326c79dfdd32d625aee71d232d685c3`. This was the first feasible
model in the requested order: the complete authorized Llama-3 8B cache and
the accepted packed signed-W4A4 runtime were both available, so Llama-2 and
Qwen3 fallback paths were not considered.

The run used job `1662352` on `vision-torralba-rtx3090`, node
`torralba-3090-1.csail.mit.edu`, with one NVIDIA GeForce RTX 3090 (SM86), UUID
`76f2cc49-6a31-24e3-23e0-f628fc728221`, and 25,296,044,032 visible bytes.
Slurm recorded `COMPLETED`, exit `0:0`, and elapsed `00:05:34` for two CPUs,
32 GiB, and one GPU. RTX 3090 was the only discovered account-allowed
Torralba class with a native accepted-extension target; V100 is incompatible,
and H100/H200 would have required the excluded PTX-JIT path. Results therefore
apply to this RTX 3090/SM86 stack and do not establish cross-GPU behavior.

## Comparable results

All quality rows evaluate the same 282 length-1024 segments: 288,486 scored
WikiText-2 tokens with 309 tail tokens truncated. Decode uses batch 1, prompt
128, output 32, one warmup, five measured repetitions, synchronized manual
greedy argmax, and cache enabled.

| Variant | WikiText-2 PPL | Decode median | ms/token | tokens/s |
| --- | ---: | ---: | ---: | ---: |
| FP16 | 6.820283 | 1,233.050 ms | 38.5328 | 25.9519 |
| packed W4A4, no rotation | 1,075.362051 | 1,698.911 ms | 53.0910 | 18.8356 |
| packed W4A4, online Hfull at FFN down projections | 101.042258 | 2,194.636 ms | 68.5824 | 14.5810 |

The affected `model.layers.0.mlp.down_proj` decode-row measurement used input
shape `[1, 14336]`, 20 warmups, 100 CUDA-event repetitions, and terminal
synchronization.

| Transform | Rotation median | Affected quantized layer median |
| --- | ---: | ---: |
| I | 0 ms | 0.184320 ms |
| Hfull | 0.578560 ms | 0.743424 ms |

Hfull-versus-I overhead is `29.179015%` end to end and `303.333326%` in the
important affected layer. The literal decision is therefore:

- A1: **proceed to Phase B** because both the 3% end-to-end and 5% affected-
  kernel thresholds are exceeded.
- A2: not applicable to this result; its `<2%` end-to-end premise is false.
- A3: not established. This run alone cannot establish A3.

The quality result is not a quality success: no-rotation W4A4 is catastrophic,
and down-projection-only Hfull improves it substantially but remains far worse
than FP16. A1 is the plan's transform-overhead gate, not an endorsement of
this minimal W4A4 recipe's accuracy.

## Correctness and scope

The sole required unquantized one-block smoke passed before timing:
rotation-inverse maximum absolute error `0.0` and folded-model equivalence
maximum absolute error `0.0`, both at tolerance `1e-12`.

Both W4A4 variants use the accepted packed signed-int4 CUTLASS runtime for all
224 Transformer q/k/v/o/gate/up/down projection linears. Activations are
dynamic symmetric per token row; weights are symmetric per output row;
accumulation is int32. Embeddings, normalization, and LM head remain FP16.
The scientific difference is limited to all 32 FFN `down_proj` inputs:
host-alias identity versus exact online full Hadamard with the matching folded
weight. This is the plan's initial-site QuaRot-style comparison, not a claim
of reproducing every global QuaRot rotation. Transform and quantized GEMM are
sequential and every result records `fusion="none"`; no fused-kernel claim is
made.

## Provenance

- Source commit/tree:
  `71ae5c823a6c4317cae74203d78716fb81c27729` /
  `3d93dfbc2aca1171855826a0fd21b8446bee9bf4`; required ancestor
  `0c4aaf075d929be2474fd271ffb8bc1244206d99` verified.
- Immutable project stage: 1,806 verified entries, independent Git metadata,
  manifest SHA-256
  `44eb32dbfec231eca78f5edacbeddf6784bc7ce994e9805469e29a5bb7660c74`.
- QuaRot/CUTLASS commits:
  `5008669b08c1f11f9b64d52d16fddd47ca754c5a` /
  `ffa34e70756b0bc744e1dfcc115b5a991a68f132`; 5,728 dependency entries
  verified immutable.
- Accepted extension SHA-256:
  `10a961e8855d7349708412aa8c21a38667fb75c94c372180dd6f992913e0a8e0`,
  native SM80/SM86 SASS and compute_80 PTX only.
- Pinned Python SHA-256:
  `fa10ee8f4c18e62cbd1e467c156a228be45138bd537b9949e66fc8e5937a018e`.
  The audit hashed exact bytes for Torch 2.8.0+cu128, Transformers 5.12.1,
  Triton 3.4.0, Safetensors 0.8.0, Accelerate 1.14.0, NumPy 1.24.4,
  Tokenizers 0.22.2, and task-local PyArrow 17.0.0 without importing them.
- Clearance/preflight SHA-256:
  `8c62ba56a3416734fdc1a4320bbab301afd120f1b2f92cc2f414de957c63791a` /
  `63d2c4915c6889c70a80973f1fc7404b2459886ec222a24e3dc5a301d78b2ff1`.
- Results/raw timing SHA-256:
  `a173ee7fc4d43ab4b33123190ea4e7893c1d5bf421587f07e2502b75fcfd61d4` /
  `7596a96d2225523b924b66b8f0c6b594bacfb187f61fbd51a6748f8c180c0381`.

The owner re-audited the full clearance before consuming the one-shot ledger.
Final counts are exactly one owner, one `salloc`, one `srun --pty`, one
scientific driver, and one terminal event. There was no `sbatch`, retry,
requeue, cancellation, second owner, duplicate stage, duplicate driver, push,
PR, merge, Phase B run, stress regime, H32/H128 run, or synthetic rerun. The
terminal event was handled, and both GPU owner and sole SSH/tmux route retired.

One provenance caveat is explicit: the result's direct Torch driver-version
field is `null` because Torch 2.8.0 lacks the private query used by the driver.
The same node was directly recorded at NVIDIA driver `580.178.04` by the prior
hardware-qualified run, but that is corroboration rather than a fresh capture
for job `1662352`. No second GPU attempt was made to repair this field.

Exact copied artifacts are under `artifacts/run-1662352/`; the prior blocked
preflight and bounded recovery evidence below remains preserved as history.

# Prior Phase A real-model baseline preflight

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
