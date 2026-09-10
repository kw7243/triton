from __future__ import annotations

import json
import math
from pathlib import Path
import unittest

from experiments.structured_rotations_v2.gate_a import execution


ROOT = Path(__file__).resolve().parents[4]


class ExecutionContractTest(unittest.TestCase):
    def test_contract_matches_quality_lane(self) -> None:
        contract = json.loads(
            (ROOT / "experiments/structured_rotations_v2/gate_a/execution_contract.json").read_text()
        )
        quality = json.loads(
            (ROOT / "experiments/structured_rotations_v2/gate_a/contract.json").read_text()
        )
        execution.validate_contract(contract, quality)
        self.assertEqual(contract["tensor_contract"]["gated_activation"], ["M", 12288])
        self.assertEqual(contract["tensor_contract"]["packed_activation"], ["M", 6144])
        self.assertEqual(
            contract["tensor_contract"]["rows"],
            {"prefill_b1_s2048": 2048, "decode_b1_ctx2048": 1,
             "decode_b8_ctx2048": 8},
        )

    def test_scope_excludes_rejected_work(self) -> None:
        contract = json.loads(
            (ROOT / "experiments/structured_rotations_v2/gate_a/execution_contract.json").read_text()
        )
        exclusions = " ".join(contract["scope"]["excluded"])
        self.assertIn("activation-quality", exclusions)
        self.assertIn("representative bridges", exclusions)
        self.assertIn("per-layer transform selection", exclusions)
        self.assertEqual(contract["budget"]["attempt_limit"], 1)
        self.assertFalse(contract["budget"]["requeue"])


class AlgebraTest(unittest.TestCase):
    def test_paley_h12_is_hadamard(self) -> None:
        rows = execution.paley_hadamard_12_rows()
        self.assertEqual((len(rows), len(rows[0])), (12, 12))
        for lhs, first in enumerate(rows):
            for rhs, second in enumerate(rows):
                dot = sum(a * b for a, b in zip(first, second))
                self.assertEqual(dot, 12 if lhs == rhs else 0)

    def test_asymmetric_signed_mapping(self) -> None:
        weights = [-7, -3, 0, 2, 7]
        for zero_point in range(16):
            for unsigned in range(16):
                signed = unsigned - 8
                for weight in weights:
                    native = signed * weight
                    corrected = native + (8 - zero_point) * weight
                    self.assertEqual(corrected, (unsigned - zero_point) * weight)

    def test_timing_summary(self) -> None:
        result = execution.summarize([1.0, 2.0, 4.0, 8.0])
        self.assertEqual(result["samples"], 4)
        self.assertTrue(math.isclose(result["median_ms"], 3.0))
        self.assertTrue(math.isclose(result["median_absolute_deviation_ms"], 1.5))


class BatchScriptTest(unittest.TestCase):
    def test_batch_script_uses_slurm_submit_dir_and_nil_safe_environment(self) -> None:
        text = (ROOT / "experiments/structured_rotations_v2/gate_a/run_execution.sbatch").read_text()
        self.assertIn("SLURM_SUBMIT_DIR", text)
        self.assertNotIn("BASH_SOURCE", text)
        self.assertNotIn("salloc", text)
        self.assertNotIn("srun", text)
        self.assertNotIn("export HOME=", text)
        self.assertIn("PYTHONNOUSERSITE=1", text)
        self.assertIn("structured-rotations-v2/envs/gate-a-quality/bin/python", text)
        self.assertNotIn("micromamba/root/envs/causal_forcing/bin/python", text)


if __name__ == "__main__":
    unittest.main()
