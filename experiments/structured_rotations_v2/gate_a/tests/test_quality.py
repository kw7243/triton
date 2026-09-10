from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.structured_rotations_v2.gate_a import quality


ROOT = Path(__file__).resolve().parents[4]
CONTRACT = ROOT / "experiments" / "structured_rotations_v2" / "gate_a" / "contract.json"


class FakeVector:
    def __init__(self, values):
        self.values = list(values)
        self.ndim = 1
        self.shape = (len(self.values),)

    def numel(self):
        return len(self.values)

    def tolist(self):
        return list(self.values)


class QualityContractTests(unittest.TestCase):
    def test_frozen_contract(self):
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        quality.validate_contract(contract)
        self.assertEqual([item["name"] for item in contract["methods"]], list(quality.METHOD_NAMES))
        self.assertEqual(contract["model"]["sampled_layers"], [0, 7, 14, 21, 28, 35])
        self.assertEqual(contract["scope"]["execution_kind"], "numerical_only")

    def test_paley_h12_is_flat_and_orthogonal(self):
        matrix = quality._paley_hadamard_12_rows()
        self.assertEqual(len(matrix), 12)
        self.assertTrue(all(len(row) == 12 for row in matrix))
        self.assertTrue(all(value in (-1, 1) for row in matrix for value in row))
        for lhs in range(12):
            for rhs in range(12):
                product = sum(matrix[lhs][index] * matrix[rhs][index] for index in range(12))
                self.assertEqual(product, 12 if lhs == rhs else 0)

    def test_massdiff_balances_membership_and_orders_tail_lanes(self):
        means = FakeVector([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0])
        tails = FakeVector([1.0, 3.0, 8.0, 2.0, 7.0, 4.0, 6.0, 5.0])
        permutation = quality.massdiff_permutation(means, tails, 4)
        self.assertEqual(sorted(permutation), list(range(8)))
        blocks = [permutation[:4], permutation[4:]]
        self.assertTrue(all(len(block) == 4 for block in blocks))
        for block in blocks:
            self.assertEqual(block, sorted(block, key=lambda index: (-tails.values[index], index)))

    def test_massdiff_is_deterministic(self):
        means = FakeVector([1.0] * 8)
        tails = FakeVector([1.0] * 8)
        first = quality.massdiff_permutation(means, tails, 2)
        second = quality.massdiff_permutation(means, tails, 2)
        self.assertEqual(first, second)
        self.assertEqual(sorted(first), list(range(8)))

    def test_batch_script_keeps_scope_and_export_boundary(self):
        script = (CONTRACT.parent / "run_quality.sbatch").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --no-requeue", script)
        self.assertNotIn("salloc", script)
        self.assertNotIn("srun", script)
        self.assertNotIn("--export=NONE", script)
        self.assertNotIn("--export=ALL", script)
        self.assertNotIn("selector", script.lower())


if __name__ == "__main__":
    unittest.main()

