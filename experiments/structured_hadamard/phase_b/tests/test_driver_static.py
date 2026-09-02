from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest

from experiments.structured_hadamard.phase_b import driver, gpu_owner, local_event_owner


class PhaseBDriverStaticTest(unittest.TestCase):

    def test_calibration_is_frozen_at_exactly_16_by_512(self):
        parser = driver._parser()
        args = parser.parse_args([
            "--source-commit", "a" * 40, "--snapshot", "/snapshot",
            "--calibration-arrow", "/train.arrow", "--calibration-sha256", "b" * 64,
            "--evaluation-arrow", "/test.arrow", "--evaluation-sha256", "c" * 64,
            "--dataset-revision", "d" * 40, "--extension", "/runtime.so",
            "--extension-sha256", "e" * 64, "--quarot-root", "/quarot",
            "--phase-a-results", "/phase-a-results.json", "--output-directory", "/output",
            "--status-path", "/status.jsonl",
        ])
        self.assertEqual((args.calibration_sequences, args.calibration_tokens), (16, 512))
        self.assertFalse(args.expand_calibration)
        self.assertEqual((args.timing_warmups, args.timing_repetitions), (20, 100))
        self.assertEqual(args.fusion, "none")

    def test_one_shot_owner_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.json"
            path.write_text(json.dumps({"owner_attempts": 0}), encoding="utf-8")
            self.assertEqual(gpu_owner._advance(path, "owner_attempts")["owner_attempts"], 1)
            with self.assertRaisesRegex(RuntimeError, "one-shot ledger refusal"):
                gpu_owner._advance(path, "owner_attempts")

    def test_local_event_owner_uses_one_controlmaster_channel_and_stable_key(self):
        args = argparse.Namespace(
            ssh="/usr/bin/ssh", control_socket=Path("/tmp/phaseb/ssh-control"),
            host="slurm-login.csail.mit.edu", remote_argv=["/pinned/python", "owner.py"],
        )
        self.assertEqual(local_event_owner.build_ssh_argv(args), [
            "/usr/bin/ssh", "-S", "/tmp/phaseb/ssh-control", "-o", "ControlMaster=no",
            "-o", "BatchMode=yes", "-tt", "slurm-login.csail.mit.edu",
            "/pinned/python", "owner.py",
        ])
        line = local_event_owner._summary(
            "phaseb-map-r1", {"event": "allocation_acquired", "job_id": "42", "partition": "p"},
        )
        self.assertIn("[key=phaseb-map-r1]", line)
        self.assertIn("job=42", line)


if __name__ == "__main__":
    unittest.main()
