from __future__ import annotations

import math
import struct
import sys
import types
import unittest
from unittest import mock

from experiments.structured_hadamard.phase_a.activation_quantizer import (A4_QMAX, A4_QMIN, _kernel_bundle,
                                                                          quantize_row_reference)
from experiments.structured_hadamard.phase_a.reference import D_FF, forward_rows, inverse_rows, transform_spec
from experiments.structured_hadamard.phase_a.triton_transform import (HFullWorkspace, _allocate_intermediate,
                                                                      apply_transform)
from experiments.structured_hadamard.phase_a.u172 import (INT8_ROW_MAJOR_SHA256, MATRIX_DIGEST, u172,
                                                           verify_orthogonality)


class TransformContractTest(unittest.TestCase):

    class _FakeStorage:

        def __init__(self, pointer):
            self.pointer = pointer

        def data_ptr(self):
            return self.pointer

    class _FakeTensor:

        ndim = 2
        shape = (1, D_FF)
        is_cuda = True
        dtype = "torch.float16"
        device = "cuda:0"

        def __init__(self, pointer, *, storage_offset=0, elements=D_FF, element_size=2):
            self._storage = TransformContractTest._FakeStorage(pointer)
            self._storage_offset = storage_offset
            self._elements = elements
            self._element_size = element_size

        def is_contiguous(self):
            return True

        def untyped_storage(self):
            return self._storage

        def storage_offset(self):
            return self._storage_offset

        def numel(self):
            return self._elements

        def element_size(self):
            return self._element_size

    class _FakeTorch:
        float32 = object()

        def __init__(self):
            self.requested_dtype = None

        def empty_like(self, example, *, dtype):
            self.requested_dtype = dtype
            return (example, dtype)

    def test_u172_digest_and_orthogonality(self):
        self.assertEqual(MATRIX_DIGEST, f"sha256:int8-row-major:{INT8_ROW_MAJOR_SHA256}")
        self.assertEqual((len(u172()), len(u172()[0])), (172, 172))
        verify_orthogonality()

    def test_contiguous_normalized_block_contracts(self):
        source = [[1.0] + [0.0] * (D_FF - 1)]
        for transform_id, block_size in (("H32", 32), ("H128", 128)):
            with self.subTest(transform_id=transform_id):
                transformed = forward_rows(source, transform_id)
                expected = 1.0 / math.sqrt(block_size)
                self.assertTrue(all(abs(value - expected) < 1e-15 for value in transformed[0][:block_size]))
                self.assertTrue(all(value == 0.0 for value in transformed[0][block_size:]))
                recovered = inverse_rows(transformed, transform_id)
                self.assertAlmostEqual(recovered[0][0], 1.0, places=12)
                self.assertLess(max(abs(value) for value in recovered[0][1:]), 1e-12)

    def test_identity_is_host_alias_without_lazy_gpu_imports(self):
        before = set(sys.modules)
        sentinel = object()
        self.assertIs(apply_transform(sentinel, "I"), sentinel)
        self.assertNotIn("torch", set(sys.modules) - before)
        self.assertNotIn("triton", set(sys.modules) - before)
        self.assertEqual(transform_spec("I").implementation, "host-no-launch")
        with self.assertRaisesRegex(ValueError, "hidden copy"):
            apply_transform(sentinel, "I", out=object())

    def test_hfull_uses_non_overflowing_fp32_intermediate(self):
        overflowing_coefficient = 64 * 1100.0
        with self.assertRaises(OverflowError):
            struct.pack("e", overflowing_coefficient)
        self.assertEqual(struct.unpack("f", struct.pack("f", overflowing_coefficient))[0], overflowing_coefficient)

        fake_torch = self._FakeTorch()
        marker = object()
        allocated = _allocate_intermediate(fake_torch, marker)
        self.assertIs(fake_torch.requested_dtype, fake_torch.float32)
        self.assertEqual(allocated, (marker, fake_torch.float32))

    def test_hfull_rejects_output_overlapping_intermediate(self):
        workspace = object.__new__(HFullWorkspace)
        workspace.shape = (1, D_FF)
        workspace.dtype = "torch.float16"
        workspace.device = "cuda:0"
        workspace.intermediate = self._FakeTensor(20_000)
        workspace.output = self._FakeTensor(50_000)
        workspace.matrix = self._FakeTensor(70_000, elements=172 * 172)
        input_tensor = self._FakeTensor(100_000)
        overlapping_output = self._FakeTensor(20_000, storage_offset=1)

        with self.assertRaisesRegex(ValueError, "overlap the intermediate"):
            workspace(input_tensor, out=overlapping_output)

    def test_hfull_rejects_input_overlapping_intermediate(self):
        workspace = object.__new__(HFullWorkspace)
        workspace.shape = (1, D_FF)
        workspace.dtype = "torch.float16"
        workspace.device = "cuda:0"
        workspace.intermediate = self._FakeTensor(20_000, element_size=4)
        workspace.output = self._FakeTensor(70_000)
        workspace.matrix = self._FakeTensor(100_000, elements=172 * 172)
        overlapping_input = self._FakeTensor(20_000, storage_offset=1)

        with self.assertRaisesRegex(ValueError, "input must not overlap the intermediate"):
            workspace(overlapping_input)

    def test_hfull_rejects_output_overlapping_matrix(self):
        workspace = object.__new__(HFullWorkspace)
        workspace.shape = (1, D_FF)
        workspace.dtype = "torch.float16"
        workspace.device = "cuda:0"
        workspace.intermediate = self._FakeTensor(20_000, element_size=4)
        workspace.output = self._FakeTensor(70_000)
        workspace.matrix = self._FakeTensor(100_000, elements=172 * 172)
        input_tensor = self._FakeTensor(200_000)
        overlapping_output = self._FakeTensor(100_000, storage_offset=1)

        with self.assertRaisesRegex(ValueError, "output must not overlap the transform matrix"):
            workspace(input_tensor, out=overlapping_output)

    def test_all_four_specs_freeze_sign_and_permutation(self):
        for transform_id in ("I", "H32", "H128", "Hfull"):
            spec = transform_spec(transform_id)
            self.assertEqual(spec.sign, "none")
            self.assertEqual(spec.permutation, "identity")
            self.assertEqual(spec.channel_order, "contiguous-natural")
        self.assertEqual((transform_spec("Hfull").K, transform_spec("Hfull").q), (172, 64))

    def test_dynamic_per_row_signed_a4_reference_semantics(self):
        row = [7.0, -7.0, 3.5, -3.5, 2.5, -2.5] + [0.0] * (D_FF - 6)
        quantized, scale = quantize_row_reference(row)
        self.assertEqual((A4_QMIN, A4_QMAX, scale), (-7, 7, 1.0))
        self.assertEqual(quantized[:6], [7.0, -7.0, 4.0, -4.0, 2.0, -2.0])
        self.assertTrue(all(A4_QMIN * scale <= value <= A4_QMAX * scale for value in quantized))

        tie_row = [1.0, 0.5] + [0.0] * (D_FF - 2)
        tie_quantized, tie_scale = quantize_row_reference(tie_row)
        self.assertEqual(tie_quantized[1], 4 * tie_scale)

    def test_triton_34_libdevice_import_and_tie_rounding_without_gpu(self):
        class Float32:

            def __init__(self, value):
                self.value = struct.unpack("f", struct.pack("f", float(value)))[0]

            def __float__(self):
                return self.value

            def __abs__(self):
                return Float32(abs(self.value))

            def __gt__(self, other):
                return self.value > float(other)

            def __mul__(self, other):
                return Float32(self.value * float(other))

            def __rmul__(self, other):
                return Float32(float(other) * self.value)

            def __truediv__(self, other):
                return Float32(self.value / float(other))

        triton = types.ModuleType("triton")
        language = types.ModuleType("triton.language")
        extra = types.ModuleType("triton.language.extra")
        libdevice = types.ModuleType("triton.language.extra.libdevice")
        triton.__path__ = []
        language.__path__ = []
        extra.__path__ = []
        triton.language = language
        language.extra = extra
        triton.jit = lambda function: function

        offsets = mock.MagicMock()
        offsets.__lt__.return_value = object()
        values = Float32(0.5)
        language.constexpr = object()
        language.float32 = object()
        language.program_id = mock.Mock(return_value=0)
        language.arange = mock.Mock(return_value=offsets)
        language.load = mock.Mock()
        language.load.return_value.to.return_value = values
        language.abs = mock.Mock(side_effect=abs)
        language.max = mock.Mock(return_value=Float32(1.0))
        language.where = mock.Mock(side_effect=lambda condition, when_true, when_false:
                                   when_true if condition else when_false)
        language.minimum = mock.Mock(side_effect=lambda left, right:
                                     left if float(left) <= float(right) else right)
        language.maximum = mock.Mock(side_effect=lambda left, right:
                                     left if float(left) >= float(right) else right)
        language.store = mock.Mock()
        libdevice.rint = mock.Mock(side_effect=lambda value: Float32(round(float(value))))

        modules = {
            "triton": triton,
            "triton.language": language,
            "triton.language.extra": extra,
            "triton.language.extra.libdevice": libdevice,
        }
        self.assertFalse(hasattr(extra, "libdevice"))
        _kernel_bundle.cache_clear()
        try:
            with mock.patch.dict(sys.modules, modules):
                kernel = _kernel_bundle()
                kernel(mock.MagicMock(), mock.MagicMock(), mock.MagicMock(), WIDTH=D_FF, BLOCK=16384,
                       QMIN=A4_QMIN, QMAX=A4_QMAX)
        finally:
            _kernel_bundle.cache_clear()
        libdevice.rint.assert_called_once()
        self.assertEqual(float(libdevice.rint.call_args.args[0]), 3.5)
        self.assertAlmostEqual(float(language.store.call_args_list[0].args[1]), 4.0 / 7.0, places=7)
        self.assertEqual(language.store.call_count, 2)

    def test_dynamic_per_row_signed_a4_zero_row_is_deterministic(self):
        quantized, scale = quantize_row_reference([0.0] * D_FF)
        self.assertEqual(scale, 1.0)
        self.assertEqual(quantized, [0.0] * D_FF)

    def test_dynamic_per_row_signed_a4_rejects_nonfinite_and_wrong_width(self):
        with self.assertRaisesRegex(ValueError, "width"):
            quantize_row_reference([0.0])
        row = [0.0] * D_FF
        row[17] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            quantize_row_reference(row)


if __name__ == "__main__":
    unittest.main()
