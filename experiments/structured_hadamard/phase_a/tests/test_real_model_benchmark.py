from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest

from experiments.structured_hadamard.phase_a.real_model import benchmark
from experiments.structured_hadamard.phase_a.real_model import gpu_owner
from experiments.structured_hadamard.phase_a.real_model import local_event_owner


def _variant(decode_ms, layer_ms):
    return {
        "decode": {"median_ms_per_output_token": decode_ms},
        "layer": {"affected_quantized_layer": {"median_ms": layer_ms}},
    }


class RealModelBenchmarkStaticTest(unittest.TestCase):

    def test_literal_decision_gates(self):
        a1 = benchmark._decision(_variant(1.0, 1.0), _variant(1.03, 1.0))
        self.assertEqual(a1["conclusion"], "A1: proceed to Phase B")
        kernel_a1 = benchmark._decision(_variant(1.0, 1.0), _variant(1.0, 1.05))
        self.assertEqual(kernel_a1["conclusion"], "A1: proceed to Phase B")
        a2 = benchmark._decision(_variant(1.0, 1.0), _variant(1.019, 1.01))
        self.assertTrue(a2["conclusion"].startswith("A2:"))
        self.assertIn("not established", a2["A3"])

    def test_owner_ledger_is_exactly_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.json"
            path.write_text(json.dumps({"owner_attempts": 0}), encoding="utf-8")
            value = gpu_owner._advance(path, "owner_attempts")
            self.assertEqual(value["owner_attempts"], 1)
            with self.assertRaisesRegex(RuntimeError, "one-shot ledger refusal"):
                gpu_owner._advance(path, "owner_attempts")

    def test_local_event_owner_builds_one_controlmaster_channel_argv(self):
        args = argparse.Namespace(
            ssh="/usr/bin/ssh", control_socket=Path("/tmp/task/ssh-control"),
            host="slurm-login.csail.mit.edu", remote_argv=["/pinned/python", "-m", "owner"],
        )
        self.assertEqual(local_event_owner.build_ssh_argv(args), [
            "/usr/bin/ssh", "-S", "/tmp/task/ssh-control", "-o", "ControlMaster=no",
            "-o", "BatchMode=yes", "-tt", "slurm-login.csail.mit.edu",
            "/pinned/python", "-m", "owner",
        ])


if __name__ == "__main__":
    unittest.main()
