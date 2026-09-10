# Third-party notice

Experiment A2 consumes the preserved `phase_a_w4a4_cuda.so` artifact built
from unchanged QuaRot `quarot/kernels/gemm.cu` and CUTLASS at QuaRot commit
`5008669b08c1f11f9b64d52d16fddd47ca754c5a`. QuaRot identifies its upstream
as [spcl/QuaRot](https://github.com/spcl/QuaRot) and is Apache-2.0 licensed.
CUTLASS retains its own upstream license. No third-party source is copied into
this repository; the exact binary SHA-256 is frozen in `execution_contract.json`.

The A2 transform and asymmetric-quantizer adapter are independent experiment
code in `execution.py`. No HARP, RUQuant, TORQ, PeRQ implementation, or old
selector source is copied. The PeRQ permutations are frozen A1 data only.
