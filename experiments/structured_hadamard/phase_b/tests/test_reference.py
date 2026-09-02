from __future__ import annotations

import math
import unittest

from experiments.structured_hadamard.phase_b import reference


class PhaseBReferenceTest(unittest.TestCase):

    def test_block_transform_inverse_and_folded_linear_equivalence(self):
        for block_size in (32, 128):
            with self.subTest(block_size=block_size):
                result = reference.block_transform_invariants(block_size)
                self.assertLessEqual(result["inverse_max_abs"], 1.0e-12)
                self.assertLessEqual(result["equivalence_max_abs"], 1.0e-12)

    def test_h32_and_h128_are_real_distinct_block_transforms(self):
        impulse = [[1.0] + [0.0] * 255]
        h32 = reference.block_hadamard_rows(impulse, 32)[0]
        h128 = reference.block_hadamard_rows(impulse, 128)[0]
        self.assertTrue(all(value == 0.0 for value in h32[32:]))
        self.assertTrue(any(value != 0.0 for value in h128[32:128]))
        self.assertAlmostEqual(sum(value * value for value in h32), 1.0)
        self.assertAlmostEqual(sum(value * value for value in h128), 1.0)

    def test_exact_block_dispatch_and_semantics(self):
        h32 = reference.block_semantics("H32", 14336, full_outer_order=7)
        h128 = reference.block_semantics("H128", 14336, full_outer_order=7)
        hfull = reference.block_semantics("Hfull", 14336, full_outer_order=7)
        self.assertEqual((h32["block_size"], h32["blocks"]), (32, 448))
        self.assertEqual((h128["block_size"], h128["blocks"]), (128, 112))
        self.assertEqual((hfull["outer_order"], hfull["inner_order"]), (7, 2048))
        self.assertEqual(h32["weight_folding"],
                         "same symmetric orthonormal block transform on each weight row")
        with self.assertRaises(reference.ReferenceError):
            reference.block_hadamard_rows([[1.0] * 63], 32)


if __name__ == "__main__":
    unittest.main()
