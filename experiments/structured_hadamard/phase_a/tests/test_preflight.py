from __future__ import annotations

import contextlib
import io
import sys
import unittest
from unittest import mock

from experiments.structured_hadamard.phase_a.preflight import (
    DEFAULT_RESULT_JSONL,
    build_plan,
    discover_code_identity,
    main,
)
from experiments.structured_hadamard.phase_a.schema import BASE_COMMIT


class PreflightTest(unittest.TestCase):

    @mock.patch("experiments.structured_hadamard.phase_a.preflight.subprocess.run")
    @mock.patch("experiments.structured_hadamard.phase_a.preflight._git")
    def test_kernel_profile_branch_discovers_clean_pinned_identity(self, git, run):
        head = "a" * 40
        git.side_effect = ["fm/structured-hadamard-phase-a-kernel-profile-r1", head, ""]
        run.return_value.returncode = 0

        self.assertEqual(discover_code_identity(), {"head": head, "dirty": False})
        run.assert_called_once_with(("git", "merge-base", "--is-ancestor", BASE_COMMIT, head), check=False)

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
            self.assertEqual(record["transform"]["fusion"], "none")
            self.assertEqual(record["artifacts"]["record_jsonl"], DEFAULT_RESULT_JSONL)
        self.assertEqual(records[2]["timing"]["composition"], "sequential-transform-then-quantize")
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
