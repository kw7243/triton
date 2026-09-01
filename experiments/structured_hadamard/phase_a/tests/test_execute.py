from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from experiments.structured_hadamard.phase_a.execute import (
    Authorization,
    CLEARANCE_SCHEMA_VERSION,
    ExecutionRefusal,
    StageIdentity,
    assemble_records,
    collect_profiles,
    execute,
    load_authorization,
    validate_raw_records,
    verify_complete_stage,
    write_output,
)
from experiments.structured_hadamard.phase_a.profiler import ProfileSummary
from experiments.structured_hadamard.phase_a.stage_repository import stage_repository


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=root, text=True, stderr=subprocess.STDOUT).strip()


def _init_repository(path: Path, *, dirty: bool = False) -> str:
    path.mkdir()
    _git(path, "init", "--quiet")
    _git(path, "config", "user.name", "Phase A Test")
    _git(path, "config", "user.email", "phase-a-test@example.invalid")
    (path / "tracked.txt").write_text("committed\n", encoding="utf-8")
    _git(path, "add", "tracked.txt")
    _git(path, "commit", "--quiet", "-m", "fixture")
    if dirty:
        (path / "untracked.py").write_text("VALUE = 1\n", encoding="utf-8")
    return _git(path, "rev-parse", "HEAD")


def _quant_config() -> dict:
    return {
        "w_bits": 4,
        "a_bits": 4,
        "w_group_size": "128",
        "a_group_size": "per-row",
        "w_symmetric": True,
        "a_symmetric": True,
        "scale_granularity": "per-group-W4;dynamic-per-row-A4",
        "clip": "none",
        "calibration_dataset": "wikitext-2-raw-v1@" + "b" * 40,
        "calibration_seed": 0,
        "calibration_rows": 8192,
    }


def _clearance(stage: StageIdentity, output: Path) -> dict:
    return {
        "schema_version": CLEARANCE_SCHEMA_VERSION,
        "scheduler_clearance": True,
        "clearance_id": "owner-clearance-001",
        "owner": "phase-a-owner",
        "owner_uid": os.geteuid(),
        "stage_root": str(stage.root),
        "source_commit": stage.head,
        "driver_commit": stage.head,
        "transform_commit": stage.head,
        "stage_manifest_sha256": stage.manifest_sha256,
        "device": "cuda:3",
        "output_directory": str(output),
        "model_revision": "a" * 40,
        "site_layer": 0,
        "quant": _quant_config(),
        "timing": {"warmup_ms": 25, "repetition_ms": 200, "outer_trials": 5},
        "clock_policy": "default-unlocked",
    }


def _write_clearance(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(0o600)


class ExecutionProvenanceTest(unittest.TestCase):

    def _clean_stage(self, root: Path) -> StageIdentity:
        source = root / "source"
        stage = root / "stage"
        _init_repository(source)
        stage_repository(source, stage)
        return verify_complete_stage(stage)

    def test_execute_subprocess_boundary_is_limited_to_stage_git_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage = self._clean_stage(root)
            clearance_path = root / "clearance.json"
            output = root / "durable-output"
            _write_clearance(clearance_path, _clearance(stage, output))
            authorization = load_authorization(clearance_path, stage)

            runtime = mock.Mock()
            runtime.hardware_identity.return_value = {
                key: "observed" for key in
                ("gpu", "compute_capability", "driver", "cuda", "torch", "triton", "clock_policy")
            }
            runtime.fixed_input.return_value = object()
            runtime.workspace.return_value = object()
            runtime.quantizer.return_value = object()
            runtime.correctness_metrics.return_value = {
                transform: {
                    "passed": True,
                    "transform_relative_error": 1e-4,
                    "transform_max_abs_error": 2e-4,
                    "quantizer_relative_error": 3e-4,
                    "quantizer_max_abs_error": 4e-4,
                    "local_nmse": 5e-3,
                    "nmse_epsilon": 1e-12,
                    "activation_absmax": 2.0,
                    "activation_rms": 0.5,
                } for transform in ("I", "Hfull")
            }
            oracle_result = {
                "inverse_relative_error": 0.0,
                "inverse_max_abs_error": 0.0,
                "local_equivalence_relative_error": 1e-8,
                "local_equivalence_max_abs_error": 1e-7,
                "passed": True,
            }
            oracle = {"results": {transform: copy.deepcopy(oracle_result) for transform in ("I", "Hfull")}}

            def fake_profile(tensor, transform_id, timing_identity, **kwargs):
                empty = (transform_id, timing_identity) == ("I", "transform-only")
                launches = 0 if transform_id == "I" else 2
                return ProfileSummary(timing_identity, 0.0 if empty else 1.0, 0.0 if empty else 2.0,
                                      0.0 if empty else 3.0, () if empty else (1.0, 2.0, 3.0), launches, 0)

            observed_commands = []
            real_popen = subprocess.Popen

            def observed_popen(args, *popen_args, **kwargs):
                command = (args, ) if isinstance(args, str) else tuple(args)
                observed_commands.append(command)
                if command[:1] != ("git", ):
                    raise AssertionError(f"unexpected subprocess command: {command}")
                return real_popen(args, *popen_args, **kwargs)

            with mock.patch("experiments.structured_hadamard.phase_a.execute.subprocess.Popen",
                            side_effect=observed_popen):
                execute(authorization, runtime_factory=mock.Mock(return_value=runtime),
                        profile_fn=fake_profile, oracle_fn=mock.Mock(return_value=oracle))

            self.assertTrue(observed_commands)
            self.assertTrue((output / "phase-a.execution.json").is_file())

    def test_complete_clean_independent_stage_and_owner_clearance_validate_without_cuda_import(self):
        before = set(sys.modules)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage = self._clean_stage(root)
            clearance_path = root / "clearance.json"
            output = root / "durable-output"
            _write_clearance(clearance_path, _clearance(stage, output))
            authorization = load_authorization(clearance_path, stage)
            self.assertEqual(authorization.device, "cuda:3")
            self.assertEqual(authorization.output_directory, output)
            self.assertRegex(authorization.config_sha256, r"^[0-9a-f]{64}$")
        imported = set(sys.modules) - before
        self.assertNotIn("torch", imported)
        self.assertNotIn("triton", imported)

    def test_stage_refuses_dirty_commit_untracked_source_and_manifest_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage = self._clean_stage(root)
            (stage.root / "tracked.txt").write_text("mutated\n", encoding="utf-8")
            with self.assertRaisesRegex(ExecutionRefusal, "changed|dirty"):
                verify_complete_stage(stage.root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            staged = root / "stage"
            _init_repository(source, dirty=True)
            stage_repository(source, staged)
            with self.assertRaisesRegex(ExecutionRefusal, "exactly the committed tracked paths"):
                verify_complete_stage(staged)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage = self._clean_stage(root)
            manifest_path = stage.root / "REPRODUCIBILITY_MANIFEST.json"
            manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaisesRegex(ExecutionRefusal, "digest mismatch"):
                verify_complete_stage(stage.root)

    def test_clearance_refuses_commit_mismatch_unresolved_pins_permissions_and_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage = self._clean_stage(root)
            path = root / "clearance.json"
            output = root / "output"
            base = _clearance(stage, output)

            cases = []
            mismatch = copy.deepcopy(base)
            mismatch["transform_commit"] = "c" * 40
            cases.append((mismatch, "transform_commit"))
            unresolved_model = copy.deepcopy(base)
            unresolved_model["model_revision"] = "main"
            cases.append((unresolved_model, "model_revision"))
            unresolved_calibration = copy.deepcopy(base)
            unresolved_calibration["quant"]["calibration_dataset"] = "UNRESOLVED"
            cases.append((unresolved_calibration, "calibration_dataset|resolved"))
            for value, message in cases:
                with self.subTest(message=message):
                    _write_clearance(path, value)
                    with self.assertRaisesRegex(ExecutionRefusal, message):
                        load_authorization(path, stage)

            _write_clearance(path, base)
            path.chmod(0o644)
            with self.assertRaisesRegex(ExecutionRefusal, "mode 0600"):
                load_authorization(path, stage)
            path.chmod(0o600)
            output.mkdir()
            with self.assertRaisesRegex(ExecutionRefusal, "overwrite"):
                load_authorization(path, stage)


class ExecutionAssemblyTest(unittest.TestCase):

    def _authorization(self, output: Path) -> Authorization:
        stage = StageIdentity(output.parent / "stage", "a" * 40, "b" * 64, "c" * 64)
        return Authorization("clearance", "owner", "d" * 64, "e" * 64, stage, "cuda:0", output,
                             "f" * 40, 0, _quant_config(),
                             {"warmup_ms": 25, "repetition_ms": 200, "outer_trials": 5}, "default")

    def test_profiler_boundary_visits_only_four_rows_and_preserves_raw_samples(self):
        calls = []

        def fake_profile(tensor, transform_id, timing_identity, **kwargs):
            calls.append((transform_id, timing_identity))
            if (transform_id, timing_identity) == ("I", "transform-only"):
                return ProfileSummary(timing_identity, 0.0, 0.0, 0.0, (), 0, 0)
            launches = 0 if transform_id == "I" else 2
            return ProfileSummary(timing_identity, 1.0, 2.0, 3.0, (1.0, 2.0, 3.0), launches, 0)

        summaries = collect_profiles(object(), object(), lambda value: value,
                                     {"warmup_ms": 25, "repetition_ms": 200, "outer_trials": 5},
                                     profile_fn=fake_profile)
        self.assertEqual(set(summaries), {
            ("transform-only", "I"), ("transform-only", "Hfull"),
            ("transform+quantize", "I"), ("transform+quantize", "Hfull"),
        })
        self.assertEqual(len(calls), 20)
        self.assertEqual(set(calls), {
            ("I", "transform-only"), ("Hfull", "transform-only"),
            ("I", "transform+quantize"), ("Hfull", "transform+quantize"),
        })
        self.assertEqual(len(summaries[("transform+quantize", "Hfull")].samples_us), 15)

    def test_record_assembly_is_validated_synthetic_non_model_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            authorization = self._authorization(output)
            oracle_result = {
                "inverse_relative_error": 0.0,
                "inverse_max_abs_error": 0.0,
                "local_equivalence_relative_error": 1e-8,
                "local_equivalence_max_abs_error": 1e-7,
                "passed": True,
            }
            oracle = {"results": {"I": copy.deepcopy(oracle_result), "Hfull": copy.deepcopy(oracle_result)}}
            runtime = {
                transform: {
                    "passed": True,
                    "transform_relative_error": 1e-4,
                    "transform_max_abs_error": 2e-4,
                    "quantizer_relative_error": 3e-4,
                    "quantizer_max_abs_error": 4e-4,
                    "local_nmse": 5e-3,
                    "nmse_epsilon": 1e-12,
                    "activation_absmax": 2.0,
                    "activation_rms": 0.5,
                } for transform in ("I", "Hfull")
            }
            profiles = {}
            for timing_identity in ("transform-only", "transform+quantize"):
                for transform in ("I", "Hfull"):
                    empty = (timing_identity, transform) == ("transform-only", "I")
                    launches = 0 if transform == "I" else 2
                    profiles[(timing_identity, transform)] = ProfileSummary(
                        timing_identity, 0.0 if empty else 1.0, 0.0 if empty else 2.0,
                        0.0 if empty else 3.0, () if empty else (1.0, 2.0, 3.0), launches, 0)
            hardware = {key: "observed" for key in
                        ("gpu", "compute_capability", "driver", "cuda", "torch", "triton", "clock_policy")}
            records, raw = assemble_records(authorization, hardware, oracle, runtime, profiles)
            self.assertEqual(len(records), 4)
            self.assertEqual(len(raw), 4)
            self.assertEqual({record["transform"]["id"] for record in records}, {"I", "Hfull"})
            self.assertTrue(all(record["transform"]["fusion"] == "none" for record in records))
            self.assertTrue(all(record["execution"]["scientific_evidence"] is False for record in records))
            self.assertTrue(all(record["metrics"]["ppl_wikitext2"] is None for record in records))
            self.assertIn("NON-MODEL/NON-PPL", records[0]["notes"])
            self.assertEqual(records[0]["execution"]["total_launches"], 0)
            self.assertEqual(records[2]["execution"]["total_launches"], 1)
            self.assertEqual(records[3]["execution"]["total_launches"], 3)
            write_output(authorization, records, raw)
            self.assertEqual(set(path.name for path in output.iterdir()), {
                "phase-a.jsonl", "phase-a.raw-samples.jsonl", "phase-a.execution.json"
            })
            self.assertEqual(len((output / "phase-a.jsonl").read_text(encoding="utf-8").splitlines()), 4)
            manifest = json.loads((output / "phase-a.execution.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["execution_config_sha256"], authorization.config_sha256)
            self.assertFalse(manifest["scientific_evidence"])
            with self.assertRaisesRegex(ExecutionRefusal, "overwrite"):
                write_output(authorization, records, raw)

    def test_raw_records_refuse_nonfinite_partial_and_duplicate_rows(self):
        base = []
        for timing_identity in ("transform-only", "transform+quantize"):
            for transform in ("I", "Hfull"):
                base.append({
                    "schema_version": "phase-a-raw-timing-v1",
                    "run_id": "run",
                    "experiment_commit": "a" * 40,
                    "transform": transform,
                    "timing_identity": timing_identity,
                    "unit": "us",
                    "samples": [] if (transform, timing_identity) == ("I", "transform-only") else [1.0],
                    "synthetic_non_model": True,
                })
        self.assertEqual(validate_raw_records(base), base)
        nonfinite = copy.deepcopy(base)
        nonfinite[1]["samples"] = [math.nan]
        with self.assertRaisesRegex(ExecutionRefusal, "finite"):
            validate_raw_records(nonfinite)
        partial = copy.deepcopy(base)
        partial[1]["samples"] = []
        with self.assertRaisesRegex(ExecutionRefusal, "partial"):
            validate_raw_records(partial)
        duplicate = copy.deepcopy(base)
        duplicate[1] = copy.deepcopy(duplicate[0])
        with self.assertRaisesRegex(ExecutionRefusal, "duplicate"):
            validate_raw_records(duplicate)


if __name__ == "__main__":
    unittest.main()
