from __future__ import annotations

import contextlib
import io
import sys
import unittest

from experiments.structured_hadamard.phase_a.preflight import DEFAULT_RESULT_JSONL, build_plan, main
from experiments.structured_hadamard.phase_a.schema import BASE_COMMIT


class PreflightTest(unittest.TestCase):

    def test_build_plan_requires_explicit_git_provenance(self):
        with self.assertRaises(TypeError):
            build_plan()

    def test_exact_unexecuted_matrix_has_no_cuda_import(self):
        before = set(sys.modules)
        records = build_plan(head=BASE_COMMIT, dirty=True)
        keys = [(record["timing"]["identity"], record["transform"]["id"]) for record in records]
        self.assertEqual(keys, [
            ("transform-only", "I"),
            ("transform-only", "Hfull"),
            ("transform+quantize", "I"),
            ("transform+quantize", "Hfull"),
        ])
        for record in records:
            self.assertEqual(record["workload"]["input_shape"], [1, 11008])
            self.assertEqual(record["workload"]["seed"], 0)
            self.assertFalse(record["execution"]["scheduler_clearance"])
            self.assertFalse(record["execution"]["scientific_evidence"])
            self.assertEqual(record["artifacts"]["record_jsonl"], DEFAULT_RESULT_JSONL)
        imported = set(sys.modules) - before
        self.assertNotIn("torch", imported)
        self.assertNotIn("triton", imported)

    def test_execute_is_unconditionally_refused(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            status = main(["--execute", "--scheduler-clearance=false"])
        self.assertEqual(status, 2)
        self.assertIn("scheduler_clearance=false", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
