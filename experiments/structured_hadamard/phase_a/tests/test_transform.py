from __future__ import annotations

import math
import sys
import unittest

from experiments.structured_hadamard.phase_a.reference import D_FF, forward_rows, inverse_rows, transform_spec
from experiments.structured_hadamard.phase_a.triton_transform import apply_transform
from experiments.structured_hadamard.phase_a.u172 import (INT8_ROW_MAJOR_SHA256, MATRIX_DIGEST, u172,
                                                           verify_orthogonality)


class TransformContractTest(unittest.TestCase):

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

    def test_all_four_specs_freeze_sign_and_permutation(self):
        for transform_id in ("I", "H32", "H128", "Hfull"):
            spec = transform_spec(transform_id)
            self.assertEqual(spec.sign, "none")
            self.assertEqual(spec.permutation, "identity")
            self.assertEqual(spec.channel_order, "contiguous-natural")
        self.assertEqual((transform_spec("Hfull").K, transform_spec("Hfull").q), (172, 64))


if __name__ == "__main__":
    unittest.main()
