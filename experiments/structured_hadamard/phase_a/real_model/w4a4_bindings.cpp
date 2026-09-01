// Minimal PyTorch binding for the unchanged QuaRot packed W4A4 kernels.
//
// This file intentionally exposes only signed-int4 quantization, packed
// int4-by-int4 GEMM, and int32 dequantization. QuaRot's KV-cache and
// FlashInfer bindings are outside this Phase A gate.

#include <torch/extension.h>

#include <gemm.h>
#include <quant.h>

namespace {

void check_cuda_contiguous_2d(const torch::Tensor &tensor, const char *name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 2, name, " must be rank 2");
}

} // namespace

torch::Tensor matmul(const torch::Tensor &a, const torch::Tensor &b) {
  check_cuda_contiguous_2d(a, "a");
  check_cuda_contiguous_2d(b, "b");
  TORCH_CHECK(a.scalar_type() == torch::kUInt8, "a must contain packed uint8 int4 values");
  TORCH_CHECK(b.scalar_type() == torch::kUInt8, "b must contain packed uint8 int4 values");
  TORCH_CHECK(a.device() == b.device(), "a and b must be on the same CUDA device");
  TORCH_CHECK(a.size(1) == b.size(1), "a and b must have the same packed K dimension");

  const uint32_t m = static_cast<uint32_t>(a.size(0));
  const uint32_t n = static_cast<uint32_t>(b.size(0));
  const uint32_t k = static_cast<uint32_t>(a.size(1) * kElementsPerVector);
  TORCH_CHECK(k % 32 == 0, "unpacked K must be a multiple of 32");

  auto output = torch::empty({m, n}, torch::dtype(torch::kInt32).device(a.device()));
  matmul_host(a.data_ptr<Int4Storage>(), b.data_ptr<Int4Storage>(), m, n, k,
              output.data_ptr<int32_t>());
  return output;
}

torch::Tensor sym_quant(const torch::Tensor &x, const torch::Tensor &scale) {
  check_cuda_contiguous_2d(x, "x");
  TORCH_CHECK(scale.is_cuda(), "scale must be a CUDA tensor");
  TORCH_CHECK(scale.is_contiguous(), "scale must be contiguous");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be float16");
  TORCH_CHECK(scale.scalar_type() == torch::kFloat16, "scale must be float16");
  TORCH_CHECK(scale.numel() == x.size(0), "scale must contain one value per row");
  TORCH_CHECK(x.device() == scale.device(), "x and scale must be on the same CUDA device");
  TORCH_CHECK(x.size(1) % 2 == 0, "x width must be even for packed int4 storage");

  const uint32_t rows = static_cast<uint32_t>(x.size(0));
  const uint32_t source_columns = static_cast<uint32_t>(x.size(1));
  const uint32_t packed_columns = cdiv(source_columns, static_cast<uint32_t>(kElementsPerVector));
  auto output = torch::empty({rows, packed_columns},
                             torch::dtype(torch::kUInt8).device(x.device()));
  sym_quant_host(reinterpret_cast<const half *>(x.data_ptr()),
                 reinterpret_cast<const half *>(scale.data_ptr()), rows,
                 source_columns, packed_columns, output.data_ptr<Int4Storage>());
  return output;
}

torch::Tensor sym_dequant(const torch::Tensor &q, const torch::Tensor &row_scale,
                          const torch::Tensor &column_scale) {
  check_cuda_contiguous_2d(q, "q");
  TORCH_CHECK(row_scale.is_cuda() && column_scale.is_cuda(),
              "scales must be CUDA tensors");
  TORCH_CHECK(row_scale.is_contiguous() && column_scale.is_contiguous(),
              "scales must be contiguous");
  TORCH_CHECK(q.scalar_type() == torch::kInt32, "q must be int32");
  TORCH_CHECK(row_scale.scalar_type() == torch::kFloat16 &&
                  column_scale.scalar_type() == torch::kFloat16,
              "scales must be float16");
  TORCH_CHECK(row_scale.numel() == q.size(0), "row_scale must contain one value per row");
  TORCH_CHECK(column_scale.numel() == q.size(1),
              "column_scale must contain one value per column");
  TORCH_CHECK(q.device() == row_scale.device() && q.device() == column_scale.device(),
              "q and scales must be on the same CUDA device");

  const uint32_t rows = static_cast<uint32_t>(q.size(0));
  const uint32_t columns = static_cast<uint32_t>(q.size(1));
  auto output = torch::empty(q.sizes(), torch::dtype(torch::kFloat16).device(q.device()));
  sym_dequant_host(q.data_ptr<int32_t>(),
                   reinterpret_cast<const half *>(row_scale.data_ptr()),
                   reinterpret_cast<const half *>(column_scale.data_ptr()), rows,
                   columns, reinterpret_cast<half *>(output.data_ptr()));
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("matmul", &matmul,
             "Packed signed-int4 A @ B^T with int32 accumulation");
  module.def("sym_quant", &sym_quant,
             "Per-row symmetric float16-to-packed-signed-int4 quantization");
  module.def("sym_dequant", &sym_dequant,
             "Row/column-scaled int32-to-float16 dequantization");
}
