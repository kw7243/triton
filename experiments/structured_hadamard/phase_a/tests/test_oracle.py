from __future__ import annotations

import unittest

from experiments.structured_hadamard.phase_a.oracle import run_oracle


class BoundedOracleTest(unittest.TestCase):

    def test_actual_width_inverse_fold_local_equivalence_and_identity(self):
        report = run_oracle(seed=0, token_rows=1, weight_rows=2)
        self.assertEqual(report["shape"], {"token_rows": 1, "weight_rows": 2, "d_ff": 11008})
        self.assertEqual(report["kind"], "CPU-VALIDATION-NOT-SCIENTIFIC-EVIDENCE")
        self.assertTrue(report["identity_aliases"])
        self.assertEqual(set(report["results"]), {"I", "H32", "H128", "Hfull"})
        self.assertTrue(all(result["passed"] for result in report["results"].values()))


if __name__ == "__main__":
    unittest.main()
