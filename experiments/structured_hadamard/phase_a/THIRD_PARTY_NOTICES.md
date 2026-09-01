# Third-party notice

`u172.py` contains a bit-packed representation of `get_had172()` from
[`kw7243/QuaRot`](https://github.com/kw7243/QuaRot) at immutable commit
`5008669b08c1f11f9b64d52d16fddd47ca754c5a`. QuaRot identifies its upstream
as [`spcl/QuaRot`](https://github.com/spcl/QuaRot) and is licensed under the
Apache License 2.0.

The Phase A code uses only that fixed sign matrix and independently implements
the transform and Triton boundary. No HARP source was used or copied.

`real_model/build_w4a4_extension.py` verifies and requests compilation of the
unchanged QuaRot `quarot/kernels/gemm.cu` and `quarot/kernels/quant.cu` files
and their interfaces at the same immutable commit. Those third-party source
files and CUTLASS are not copied into this repository. The local
`real_model/w4a4_bindings.cpp` file is a minimal PyTorch compatibility binding
for that Apache-2.0 code; it deliberately excludes QuaRot's KV-cache and
FlashInfer bindings.

`real_model/runtime.py` reads the exact unchanged
`quarot/functional/hadamard.py` from the independently staged QuaRot commit to
obtain the Llama-3 `14336 = 28 x 512` full-Hadamard outer matrix. That source
is not copied here. The runtime applies the factorization independently with
ordinary Torch operations at the FFN down-projection boundary only.
