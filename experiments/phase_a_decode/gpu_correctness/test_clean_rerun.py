"""Executable CPU-only tests for the clean-rerun preparation guarantees."""

from __future__ import annotations

import base64
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from experiments.phase_a_decode.gpu_correctness.prepare_clean_rerun import (
    EXECUTABLE_INPUTS,
    LEDGER_NAME,
    MANIFEST_NAME,
    MINIMUM_VRAM_BYTES,
    PREPARER,
    PREFLIGHT_LABELS,
    REQUEST_NAME,
    STAGE_ATTESTATION_NAME,
    STAGE_VERIFICATION_NAME,
    PreparationError,
    clone_standalone_source,
    create_stage_only_attestation,
    prepare_frozen_stage,
    read_ledger,
    run_cpu_preflight,
    sha256,
    submit_once,
    validate_scheduler_contract,
    verify_self_contained_repo,
)
from experiments.phase_a_decode.gpu_correctness.protocol import (
    ABS_TOL,
    REL_TOL,
    CorrectnessArguments,
    chunks_per_record,
    contract_snapshot,
    validate_correctness_arguments,
)
from experiments.phase_a_decode.gpu_correctness.result_protocol import (
    atomic_write_json,
    classify_result,
    finalize_result,
    main as result_protocol_main,
    mark_payload_complete,
    record_boundary,
)
from experiments.phase_a_decode.gpu_correctness.validate_environment import (
    validated_stage_untracked,
)


def run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr)
    return result.stdout.strip()


def make_writable(root: Path) -> None:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if not path.is_symlink():
            path.chmod(0o755 if path.is_dir() else 0o644)
    root.chmod(0o755)


def passed_correctness(bf16_supported: bool = False) -> dict[str, object]:
    records = []
    dtypes = ["float16"] + (["bfloat16"] if bf16_supported else [])
    for dtype in dtypes:
        for secondary_size in (96, 192):
            for input_kind in ("exhaustive", "random"):
                for config in ("b256-w4", "b512-w8"):
                    records.append(
                        {
                            "dtype": dtype,
                            "S": secondary_size,
                            "input": input_kind,
                            "config": config,
                            "chunks": chunks_per_record(secondary_size, input_kind),
                            "checks": {
                                "finite": True,
                                "J_bitwise_gather": True,
                                "F_H_axis_bitwise_J": True,
                                "F_H_vs_J_tolerance": True,
                                "F_H_vs_quantized_oracle_tolerance": True,
                                "F_H_vs_float32_oracle_tolerance": dtype == "float16",
                            },
                            "metrics": {
                                f"{variant}_vs_{reference}": {
                                    "max_abs": 0.0,
                                    "relative_fro": 0.0,
                                }
                                for variant in ("F", "H")
                                for reference in ("J", "oracle", "quantized_oracle")
                            },
                        }
                    )
    return {
        "schema": "vq-phase-a-correctness/v2",
        "status": "passed",
        "scientific_classification": "PASS",
        "mode": "correctness-only",
        "timing_executed": False,
        "tuning_executed": False,
        "cuda_graphs_executed": False,
        "bf16_supported": bf16_supported,
        "tolerances": {"absolute": ABS_TOL, "relative_frobenius": REL_TOL},
        "records": records,
    }


class ProtocolTests(unittest.TestCase):
    def test_fixed_correctness_contract_and_counts(self) -> None:
        validate_correctness_arguments(
            CorrectnessArguments(True, 0, (96, 192), 2, 8, 128)
        )
        snapshot = contract_snapshot(bf16_supported=True)
        self.assertEqual(snapshot["tolerances"], {"absolute": 4e-3, "relative_frobenius": 1e-3})
        self.assertEqual(snapshot["records_per_dtype"], 8)
        self.assertEqual(snapshot["expected_records"], 16)
        self.assertEqual(snapshot["chunks"]["S96-exhaustive"], 36_864)
        self.assertEqual(snapshot["chunks"]["S192-exhaustive"], 73_728)
        self.assertEqual(snapshot["chunks"]["S96-random"], 65_584)
        self.assertIs(snapshot["timing_allowed"], False)

    def test_workload_drift_and_timing_mode_fail_closed(self) -> None:
        invalid = (
            CorrectnessArguments(False, 0, (96, 192), 2, 8, 128),
            CorrectnessArguments(True, 1, (96, 192), 2, 8, 128),
            CorrectnessArguments(True, 0, (192, 96), 2, 8, 128),
            CorrectnessArguments(True, 0, (96, 192), 1, 8, 128),
            CorrectnessArguments(True, 0, (96, 192), 2, 4, 128),
            CorrectnessArguments(True, 0, (96, 192), 2, 8, 64),
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                validate_correctness_arguments(arguments)


class ResultClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_complete_pass_requires_payload_completion(self) -> None:
        digest = "a" * 64
        atomic_write_json(self.root / "correctness.json", passed_correctness())
        atomic_write_json(
            self.root / "run_metadata.json",
            {
                "schema": "vq-phase-a-run-metadata/v2",
                "status": "comparison-passed",
                "correctness": "PASS",
                "run_complete": False,
                "timing_executed": False,
                "tuning_executed": False,
                "cuda_graphs_executed": False,
                "launch_manifest_sha256": digest,
            },
        )
        mark_payload_complete(self.root, digest)
        result = classify_result(self.root, 0)
        self.assertEqual(result["classification"], "PASS")
        self.assertEqual(result["scientific_comparison"], "PASS")

    def test_missing_completion_metadata_is_no_result(self) -> None:
        atomic_write_json(self.root / "correctness.json", passed_correctness())
        record_boundary(
            self.root,
            "payload_complete",
            launch_manifest_sha256="a" * 64,
        )
        result = classify_result(self.root, 0)
        self.assertEqual(result["classification"], "NO RESULT")
        self.assertEqual(result["scientific_comparison"], "PASS")

    def test_post_write_failure_is_no_result_with_comparison_pass_preserved(self) -> None:
        atomic_write_json(self.root / "correctness.json", passed_correctness())
        record_boundary(self.root, "comparison_output_written")
        manifest = finalize_result(self.root, 17, "/source", "/stage", "a" * 64)
        self.assertEqual(manifest["classification"], "NO RESULT")
        self.assertEqual(manifest["scientific_comparison"], "PASS")
        self.assertEqual(manifest["failure_phase"], "after_artifact_writing")

    def test_compile_or_launch_failure_is_no_result(self) -> None:
        atomic_write_json(
            self.root / "correctness.json",
            {
                "status": "failed",
                "scientific_classification": "NO RESULT",
                "failure_kind": "infrastructure",
                "failure_phase": "kernel_compile_or_launch",
            },
        )
        record_boundary(self.root, "kernel_compile_or_launch")
        result = classify_result(self.root, 1)
        self.assertEqual(result["classification"], "NO RESULT")
        self.assertEqual(result["scientific_comparison"], "NO RESULT")

    def test_declared_numerical_comparison_failure_is_fail(self) -> None:
        atomic_write_json(
            self.root / "correctness.json",
            {
                "status": "failed",
                "scientific_classification": "FAIL",
                "failure_kind": "numerical_comparison",
                "failure_phase": "inside_comparison",
                "failed_check": "finite",
            },
        )
        record_boundary(self.root, "inside_comparison")
        result = classify_result(self.root, 1)
        self.assertEqual(result["classification"], "FAIL")
        self.assertEqual(result["scientific_comparison"], "FAIL")

    def test_unrecognized_assertion_cannot_become_scientific_fail(self) -> None:
        atomic_write_json(
            self.root / "correctness.json",
            {
                "status": "failed",
                "scientific_classification": "FAIL",
                "failure_kind": "numerical_comparison",
                "failure_phase": "inside_comparison",
                "failed_check": "unknown_assertion",
            },
        )
        record_boundary(self.root, "inside_comparison")
        self.assertEqual(classify_result(self.root, 1)["classification"], "NO RESULT")

    def test_incomplete_pass_artifact_is_no_result_not_comparison_pass(self) -> None:
        artifact = passed_correctness()
        artifact["records"] = artifact["records"][:-1]
        atomic_write_json(self.root / "correctness.json", artifact)
        record_boundary(self.root, "payload_complete")
        result = classify_result(self.root, 0)
        self.assertEqual(result["classification"], "NO RESULT")
        self.assertEqual(result["scientific_comparison"], "NO RESULT")
        return_code = result_protocol_main(
            [
                "finalize",
                "--result",
                str(self.root),
                "--payload-exit-code",
                "0",
                "--source",
                "/source",
                "--stage",
                "/stage",
                "--manifest-sha256",
                "a" * 64,
            ]
        )
        self.assertEqual(return_code, 1)

    def test_timing_artifact_prevents_payload_completion(self) -> None:
        digest = "a" * 64
        atomic_write_json(self.root / "correctness.json", passed_correctness())
        atomic_write_json(
            self.root / "run_metadata.json",
            {"launch_manifest_sha256": digest, "run_complete": False},
        )
        (self.root / "timings.csv").write_text("forbidden\n")
        with self.assertRaisesRegex(RuntimeError, "forbidden timing artifacts"):
            mark_payload_complete(self.root, digest)
        state = json.loads((self.root / "execution_state.json").read_text())
        self.assertEqual(state["boundary"], "after_artifact_writing")


class SourceMutationTests(unittest.TestCase):
    def test_preparer_entrypoint_does_not_mutate_source_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "verifier"
            source.mkdir()
            module_root = Path(__file__).resolve().parent
            for name in ("prepare_clean_rerun.py", "protocol.py", "result_protocol.py"):
                shutil.copy2(module_root / name, source / name)

            def inventory() -> dict[str, tuple[str, int, int, str | None]]:
                records = {}
                for path in sorted(
                    source.rglob("*"), key=lambda item: os.fsencode(str(item))
                ):
                    info = path.lstat()
                    records[str(path.relative_to(source))] = (
                        "directory" if path.is_dir() else "file",
                        stat.S_IMODE(info.st_mode),
                        info.st_size,
                        None if path.is_dir() else sha256(path),
                    )
                return records

            before = inventory()
            environment = os.environ.copy()
            environment.pop("PYTHONDONTWRITEBYTECODE", None)
            completed = subprocess.run(
                [sys.executable, str(source / "prepare_clean_rerun.py"), "--help"],
                cwd=source,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(inventory(), before)


class PreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "compute-native-vq"
        for name in (
            "worktrees",
            "standalone-sources",
            "staging",
            "results",
            "tools",
            "run-state",
        ):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.helper = self.root / "tools" / "stage_and_run.sh"
        self.helper.write_text("#!/bin/sh\nexit 0\n")
        self.helper.chmod(0o755)
        self.wrapper = self.root / "tools" / "rsync-wrapper" / "rsync"
        self.wrapper.parent.mkdir()
        self.wrapper.write_text("#!/bin/sh\nexec /usr/bin/rsync --info=progress2 \"$@\"\n")
        self.wrapper.chmod(0o500)
        self.helper_patch = patch(
            "experiments.phase_a_decode.gpu_correctness.prepare_clean_rerun.REQUIRED_STAGE_HELPER",
            self.helper,
        )
        self.helper_patch.start()
        self.addCleanup(self.helper_patch.stop)
        self.helper_hash_patch = patch(
            "experiments.phase_a_decode.gpu_correctness.prepare_clean_rerun."
            "REQUIRED_STAGE_HELPER_SHA256",
            sha256(self.helper),
        )
        self.helper_hash_patch.start()
        self.addCleanup(self.helper_hash_patch.stop)
        self.wrapper_path_patch = patch(
            "experiments.phase_a_decode.gpu_correctness.prepare_clean_rerun."
            "REQUIRED_RSYNC_WRAPPER",
            self.wrapper,
        )
        self.wrapper_path_patch.start()
        self.addCleanup(self.wrapper_path_patch.stop)
        self.wrapper_hash_patch = patch(
            "experiments.phase_a_decode.gpu_correctness.prepare_clean_rerun."
            "REQUIRED_RSYNC_WRAPPER_SHA256",
            sha256(self.wrapper),
        )
        self.wrapper_hash_patch.start()
        self.addCleanup(self.wrapper_hash_patch.stop)
        self.source = self.root / "worktrees" / "source"
        self.source.mkdir()
        run_git(self.source, "init")
        run_git(self.source, "config", "user.name", "Test")
        run_git(self.source, "config", "user.email", "test@example.invalid")
        for relative in EXECUTABLE_INPUTS:
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".sbatch":
                path.write_text("#!/bin/bash\nset -euo pipefail\n")
                path.chmod(0o755)
            else:
                path.write_text(f"# fixture {relative}\n")
        run_git(self.source, "add", ".")
        run_git(self.source, "commit", "-m", "fixture")
        self.commit = run_git(self.source, "rev-parse", "HEAD")
        self.tree = run_git(self.source, "rev-parse", "HEAD^{tree}")
        self.standalone = self.root / "standalone-sources" / "source"
        clone_standalone_source(self.source, self.standalone, self.commit, self.root)

    def tearDown(self) -> None:
        make_writable(self.root)
        self.temporary.cleanup()

    def make_stage_inputs(self, *, stage_only: bool = False) -> tuple[Path, Path, Path]:
        stage = self.root / "staging" / "stage-code"
        shutil.copytree(self.standalone, stage, symlinks=True)
        atomic_write_json(
            stage / "REPRODUCIBILITY_METADATA.json",
            {
                "command": "" if stage_only else f"{sys.executable} prepare_clean_rerun.py freeze-stage",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "cwd": str(self.standalone),
                "git_commit_full": self.commit,
                "git_commit_short": self.commit[:10],
                "git_status_short": "",
                "hostname": "fixture.invalid",
                "source_repo": str(self.standalone),
                "staged_repo": str(stage),
            },
        )
        result = self.root / "results" / "attempt"
        result.mkdir(mode=0o700)
        preflight = result / "cpu_preflight.json"
        atomic_write_json(
            preflight,
            {
                "schema": "vq-phase-a-cpu-preflight/v2",
                "status": "passed",
                "repo": str(self.standalone),
                "expected_commit": self.commit,
                "git": {"head": self.commit, "tree": self.tree},
                "environment": {
                    "python": "fixture",
                    "torch": "fixture",
                    "triton": "fixture",
                    "matplotlib": "fixture",
                    "cuda_visible_devices": "",
                    "cuda_available": False,
                },
                "commands": [
                    {
                        "label": label,
                        "argv": [sys.executable],
                        "returncode": 0,
                        "stdout": "",
                        "stderr": "",
                    }
                    for label in PREFLIGHT_LABELS
                ],
                "input_hashes": {
                    relative: sha256(self.standalone / relative)
                    for relative in EXECUTABLE_INPUTS
                },
                "contract": contract_snapshot(bf16_supported=None),
            },
        )
        evidence = result / "scheduler_preflight.txt"
        evidence.write_text("fixture current scheduler evidence\n")
        scheduler = result / "scheduler_selection.json"
        atomic_write_json(
            scheduler,
            {
                "schema": "vq-phase-a-scheduler-selection/v1",
                "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                "account": "fixture-account",
                "qos": "fixture-qos",
                "partition": "fixture-torralba-partition",
                "torralba_only": True,
                "selection_rationale": "fixture GPU satisfies the predeclared constraints",
                "resources": {
                    "nodes": 1,
                    "tasks": 1,
                    "cpus": 4,
                    "gpus": 1,
                    "memory_gib": 16,
                    "time_minutes": 15,
                },
                "adequacy": {
                    "advertised_gpu_names": ["Fixture GPU"],
                    "minimum_vram_bytes": MINIMUM_VRAM_BYTES,
                    "cuda_required": True,
                    "triton_required": True,
                    "exactly_one_visible_gpu": True,
                    "bf16_policy": "required-when-supported",
                },
                "selection_evidence": {
                    "file": evidence.name,
                    "sha256": sha256(evidence),
                },
            },
        )
        if stage_only:
            log = self.root / "run-state" / "stage-only.log"
            log.write_text(
                "\n".join(
                    (
                        "Staging repository:",
                        f"  source: {self.standalone}",
                        f"  target: {stage}",
                        f"Staging complete: {stage}",
                        str(stage),
                        "",
                    )
                )
            )
            started = datetime.now(timezone.utc)
            finished = started + timedelta(seconds=1)
            create_stage_only_attestation(
                self.standalone,
                stage,
                result / STAGE_ATTESTATION_NAME,
                self.helper,
                sha256(self.helper),
                self.wrapper,
                sha256(self.wrapper),
                log,
                sha256(log),
                0,
                started.isoformat(),
                finished.isoformat(),
                1.0,
                self.root,
            )
        return stage, result, preflight

    def stage_only_invocation(
        self, stage: Path, result: Path, preflight: Path
    ) -> dict[str, object]:
        scheduler = result / "scheduler_selection.json"
        return {
            "schema": "vq-phase-a-stage-only-continuation-invocation/v1",
            "argv": [
                sys.executable,
                PREPARER,
                "freeze-stage",
                "--stage-only-continuation",
                "--stage-only-attestation",
                str(result / STAGE_ATTESTATION_NAME),
                "--source",
                str(self.standalone),
                "--result",
                str(result),
                "--cpu-preflight",
                str(preflight),
                "--scheduler-contract",
                str(scheduler),
                "--commit",
                self.commit,
                "--tree",
                self.tree,
                "--python",
                sys.executable,
                "--approved-root",
                str(self.root),
            ],
            "cwd": str(stage),
            "environment": {
                "RESEARCH_REPRO_STAGED_DIR": str(stage),
                "RESEARCH_REPRO_SOURCE_REPO": str(self.standalone),
            },
        }

    def prepare(self) -> tuple[Path, Path, dict[str, object]]:
        stage, result, preflight = self.make_stage_inputs()
        prepared = prepare_frozen_stage(
            self.standalone,
            stage,
            result,
            preflight,
            result / "scheduler_selection.json",
            self.commit,
            self.tree,
            sys.executable,
            self.root,
        )
        return stage, result, prepared

    def prepare_stage_only(self) -> tuple[Path, Path, dict[str, object]]:
        stage, result, preflight = self.make_stage_inputs(stage_only=True)
        prepared = prepare_frozen_stage(
            self.standalone,
            stage,
            result,
            preflight,
            result / "scheduler_selection.json",
            self.commit,
            self.tree,
            sys.executable,
            self.root,
            self.stage_only_invocation(stage, result, preflight),
            result / STAGE_ATTESTATION_NAME,
        )
        return stage, result, prepared

    def test_external_worktree_git_pointer_is_rejected(self) -> None:
        bad_stage = self.root / "staging" / "external-pointer"
        bad_stage.mkdir()
        (bad_stage / ".git").write_text(f"gitdir: {self.source / '.git'}\n")
        with self.assertRaisesRegex(PreparationError, "must be a directory"):
            verify_self_contained_repo(bad_stage, self.commit)

    def test_cpu_preflight_rejects_non_detached_source_before_commands(self) -> None:
        output = self.root / "results" / "rejected-preflight.json"
        with self.assertRaises(PreparationError):
            run_cpu_preflight(self.source, output, self.commit)
        report = json.loads(output.read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["commands"], [])

    def test_stage_is_self_contained_read_only_and_manifest_bound(self) -> None:
        stage, result, prepared = self.prepare()
        self.assertTrue((stage / ".git").is_dir())
        self.assertEqual(stat.S_IMODE(stage.stat().st_mode), 0o555)
        with self.assertRaises(PermissionError):
            (stage / EXECUTABLE_INPUTS[0]).write_text("mutation\n")
        manifest_path = result / MANIFEST_NAME
        request = json.loads((result / REQUEST_NAME).read_text())
        self.assertEqual(request["launch_manifest_sha256"], sha256(manifest_path))
        self.assertIn("--no-requeue", request["sbatch_argv"])
        self.assertFalse(any(item.startswith("--export") for item in request["sbatch_argv"]))
        self.assertEqual(
            request["sbatch_environment"]["PHASE_A_MANIFEST_SHA256"],
            sha256(manifest_path),
        )
        self.assertEqual(prepared["manifest_sha256"], sha256(manifest_path))
        records = read_ledger(result / LEDGER_NAME)
        self.assertEqual([record["event"] for record in records], ["prepared"])
        self.assertEqual(records[0]["payload"]["sbatch_invocations"], 0)

    def test_stage_only_continuation_binds_exact_split_boundary_once(self) -> None:
        stage, result, _ = self.prepare_stage_only()
        verification_path = stage / STAGE_VERIFICATION_NAME
        verification = json.loads(verification_path.read_text())
        manifest = json.loads((result / MANIFEST_NAME).read_text())
        self.assertEqual(stat.S_IMODE(verification_path.stat().st_mode), 0o444)
        self.assertEqual(verification["mode"], "stage-only-continuation")
        self.assertEqual(verification["helper_metadata"]["command"], "")
        self.assertEqual(
            verification["separate_freeze_invocation"]["environment"],
            {
                "RESEARCH_REPRO_STAGED_DIR": str(stage),
                "RESEARCH_REPRO_SOURCE_REPO": str(self.standalone),
            },
        )
        self.assertEqual(
            manifest["stage_boundary"],
            {
                "mode": "stage-only-continuation",
                "verification_file": STAGE_VERIFICATION_NAME,
                "verification_sha256": sha256(verification_path),
                "helper_attestation_file": STAGE_ATTESTATION_NAME,
                "helper_attestation_sha256": sha256(
                    result / STAGE_ATTESTATION_NAME
                ),
            },
        )
        self.assertEqual(
            validated_stage_untracked(manifest, stage),
            ("REPRODUCIBILITY_METADATA.json", STAGE_VERIFICATION_NAME),
        )
        with self.assertRaisesRegex(PreparationError, "already bound"):
            prepare_frozen_stage(
                self.standalone,
                stage,
                result,
                result / "cpu_preflight.json",
                result / "scheduler_selection.json",
                self.commit,
                self.tree,
                sys.executable,
                self.root,
                verification["separate_freeze_invocation"],
                result / STAGE_ATTESTATION_NAME,
            )

    def test_runtime_validation_rejects_stage_binding_hash_mismatch(self) -> None:
        stage, result, _ = self.prepare_stage_only()
        manifest = json.loads((result / MANIFEST_NAME).read_text())
        manifest["stage_boundary"]["verification_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "verification hash mismatch"):
            validated_stage_untracked(manifest, stage)

    def test_stage_only_continuation_binding_is_stage_scoped(self) -> None:
        stage, result, _ = self.prepare_stage_only()
        verification = json.loads((stage / STAGE_VERIFICATION_NAME).read_text())
        second_result = self.root / "results" / "second-attempt"
        second_result.mkdir(mode=0o700)
        for name in (
            "cpu_preflight.json",
            "scheduler_preflight.txt",
            "scheduler_selection.json",
            STAGE_ATTESTATION_NAME,
        ):
            shutil.copy2(result / name, second_result / name)
        with self.assertRaisesRegex(PreparationError, "already bound"):
            prepare_frozen_stage(
                self.standalone,
                stage,
                second_result,
                second_result / "cpu_preflight.json",
                second_result / "scheduler_selection.json",
                self.commit,
                self.tree,
                sys.executable,
                self.root,
                verification["separate_freeze_invocation"],
                second_result / STAGE_ATTESTATION_NAME,
            )

    def test_stage_only_continuation_rejects_changed_helper_bytes(self) -> None:
        stage, result, preflight = self.make_stage_inputs(stage_only=True)
        self.helper.write_text("#!/bin/sh\nexit 1\n")
        with self.assertRaisesRegex(PreparationError, "stage helper SHA-256 mismatch"):
            prepare_frozen_stage(
                self.standalone,
                stage,
                result,
                preflight,
                result / "scheduler_selection.json",
                self.commit,
                self.tree,
                sys.executable,
                self.root,
                self.stage_only_invocation(stage, result, preflight),
                result / STAGE_ATTESTATION_NAME,
            )

    def test_stage_only_continuation_rejects_nonempty_unrelated_command(self) -> None:
        stage, result, preflight = self.make_stage_inputs(stage_only=True)
        metadata_path = stage / "REPRODUCIBILITY_METADATA.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["command"] = "python unrelated.py"
        atomic_write_json(metadata_path, metadata)
        with self.assertRaisesRegex(PreparationError, "requires an empty helper command"):
            prepare_frozen_stage(
                self.standalone,
                stage,
                result,
                preflight,
                result / "scheduler_selection.json",
                self.commit,
                self.tree,
                sys.executable,
                self.root,
                self.stage_only_invocation(stage, result, preflight),
                result / STAGE_ATTESTATION_NAME,
            )

    def test_stage_only_continuation_rejects_invocation_mismatch(self) -> None:
        stage, result, preflight = self.make_stage_inputs(stage_only=True)
        invocation = self.stage_only_invocation(stage, result, preflight)
        invocation["environment"]["RESEARCH_REPRO_SOURCE_REPO"] = "/wrong/source"
        with self.assertRaisesRegex(PreparationError, "environment mismatch"):
            prepare_frozen_stage(
                self.standalone,
                stage,
                result,
                preflight,
                result / "scheduler_selection.json",
                self.commit,
                self.tree,
                sys.executable,
                self.root,
                invocation,
                result / STAGE_ATTESTATION_NAME,
            )

    def test_stage_only_continuation_rejects_symlinked_metadata(self) -> None:
        stage, result, preflight = self.make_stage_inputs(stage_only=True)
        metadata_path = stage / "REPRODUCIBILITY_METADATA.json"
        target = self.root / "metadata-target.json"
        target.write_bytes(metadata_path.read_bytes())
        metadata_path.unlink()
        metadata_path.symlink_to(target)
        with self.assertRaisesRegex(PreparationError, "metadata must not be a symlink"):
            prepare_frozen_stage(
                self.standalone,
                stage,
                result,
                preflight,
                result / "scheduler_selection.json",
                self.commit,
                self.tree,
                sys.executable,
                self.root,
                self.stage_only_invocation(stage, result, preflight),
                result / STAGE_ATTESTATION_NAME,
            )

    def test_stage_only_continuation_rejects_partial_stage(self) -> None:
        stage, result, preflight = self.make_stage_inputs(stage_only=True)
        (stage / EXECUTABLE_INPUTS[0]).unlink()
        with self.assertRaisesRegex(PreparationError, "repository status entries"):
            prepare_frozen_stage(
                self.standalone,
                stage,
                result,
                preflight,
                result / "scheduler_selection.json",
                self.commit,
                self.tree,
                sys.executable,
                self.root,
                self.stage_only_invocation(stage, result, preflight),
                result / STAGE_ATTESTATION_NAME,
            )

    def test_submit_once_records_one_response_and_refuses_retry(self) -> None:
        stage, result, _ = self.prepare()
        calls = []

        def fake_runner(
            argv: list[str], cwd: Path, environment: dict[str, str]
        ) -> subprocess.CompletedProcess[str]:
            calls.append((argv, cwd, environment))
            return subprocess.CompletedProcess(argv, 0, "424242\n", "fixture\n")

        previous = Path.cwd()
        try:
            os.chdir(stage)
            job_id = submit_once(
                result / MANIFEST_NAME,
                result / REQUEST_NAME,
                result / LEDGER_NAME,
                runner=fake_runner,
            )
            self.assertEqual(job_id, "424242")
            with self.assertRaisesRegex(PreparationError, "retry is forbidden"):
                submit_once(
                    result / MANIFEST_NAME,
                    result / REQUEST_NAME,
                    result / LEDGER_NAME,
                    runner=fake_runner,
                )
        finally:
            os.chdir(previous)
        self.assertEqual(len(calls), 1)
        records = read_ledger(result / LEDGER_NAME)
        self.assertEqual(
            [record["event"] for record in records],
            ["prepared", "submission_started", "submission_finished", "accepted"],
        )
        self.assertEqual(records[-1]["payload"]["job_id"], "424242")

    def test_failed_submission_consumes_attempt_and_preserves_response_bytes(self) -> None:
        stage, result, _ = self.prepare()
        calls = []

        def fake_runner(
            argv: list[str], cwd: Path, environment: dict[str, str]
        ) -> subprocess.CompletedProcess[bytes]:
            calls.append((argv, cwd, environment))
            return subprocess.CompletedProcess(argv, 9, b"", b"exact failure bytes\n")

        previous = Path.cwd()
        try:
            os.chdir(stage)
            with self.assertRaisesRegex(PreparationError, "retry is forbidden"):
                submit_once(
                    result / MANIFEST_NAME,
                    result / REQUEST_NAME,
                    result / LEDGER_NAME,
                    runner=fake_runner,
                )
            with self.assertRaisesRegex(PreparationError, "retry is forbidden"):
                submit_once(
                    result / MANIFEST_NAME,
                    result / REQUEST_NAME,
                    result / LEDGER_NAME,
                    runner=fake_runner,
                )
        finally:
            os.chdir(previous)
        self.assertEqual(len(calls), 1)
        records = read_ledger(result / LEDGER_NAME)
        self.assertEqual(
            [record["event"] for record in records],
            ["prepared", "submission_started", "submission_finished"],
        )
        encoded = records[-1]["payload"]["stderr_base64"]
        self.assertEqual(base64.b64decode(encoded), b"exact failure bytes\n")

    def test_ambient_sbatch_variables_stop_before_attempt_is_consumed(self) -> None:
        stage, result, _ = self.prepare()
        calls = []

        def fake_runner(
            argv: list[str], cwd: Path, environment: dict[str, str]
        ) -> subprocess.CompletedProcess[str]:
            calls.append((argv, cwd, environment))
            return subprocess.CompletedProcess(argv, 0, "424242\n", "")

        previous = Path.cwd()
        try:
            os.chdir(stage)
            with patch.dict(os.environ, {"SBATCH_EXPORT": "NONE"}):
                with self.assertRaisesRegex(PreparationError, "ambient SBATCH_\\*"):
                    submit_once(
                        result / MANIFEST_NAME,
                        result / REQUEST_NAME,
                        result / LEDGER_NAME,
                        runner=fake_runner,
                    )
        finally:
            os.chdir(previous)
        self.assertEqual(calls, [])
        self.assertFalse((result / ".launch.lock").exists())
        self.assertEqual(
            [record["event"] for record in read_ledger(result / LEDGER_NAME)],
            ["prepared"],
        )

    def test_stage_inventory_drift_stops_before_submission(self) -> None:
        stage, result, _ = self.prepare()
        make_writable(stage)
        (stage / EXECUTABLE_INPUTS[0]).write_text("tampered\n")
        calls = []

        def fake_runner(
            argv: list[str], cwd: Path, environment: dict[str, str]
        ) -> subprocess.CompletedProcess[str]:
            calls.append((argv, cwd, environment))
            return subprocess.CompletedProcess(argv, 0, "424242\n", "")

        previous = Path.cwd()
        try:
            os.chdir(stage)
            with self.assertRaisesRegex(PreparationError, "inventory changed"):
                submit_once(
                    result / MANIFEST_NAME,
                    result / REQUEST_NAME,
                    result / LEDGER_NAME,
                    runner=fake_runner,
                )
        finally:
            os.chdir(previous)
        self.assertEqual(calls, [])
        self.assertEqual([record["event"] for record in read_ledger(result / LEDGER_NAME)], ["prepared"])

    def test_stale_scheduler_selection_fails_before_stage_freeze(self) -> None:
        _, result, _ = self.make_stage_inputs()
        contract_path = result / "scheduler_selection.json"
        contract = json.loads(contract_path.read_text())
        contract["captured_at_utc"] = (
            datetime.now(timezone.utc) - timedelta(minutes=16)
        ).isoformat()
        with self.assertRaisesRegex(PreparationError, "within 15 minutes"):
            validate_scheduler_contract(contract, result)

    def test_scheduler_selection_rejects_boolean_and_float_resources(self) -> None:
        _, result, _ = self.make_stage_inputs()
        contract = json.loads((result / "scheduler_selection.json").read_text())
        for key, value in (("nodes", True), ("cpus", 4.0)):
            rejected = json.loads(json.dumps(contract))
            rejected["resources"][key] = value
            with self.subTest(key=key, value=value):
                with self.assertRaisesRegex(
                    PreparationError, "non-boolean integers"
                ):
                    validate_scheduler_contract(rejected, result)
        rejected = json.loads(json.dumps(contract))
        rejected["adequacy"]["minimum_vram_bytes"] = True
        with self.assertRaisesRegex(PreparationError, "at least"):
            validate_scheduler_contract(rejected, result)

    def test_incomplete_reproducibility_metadata_fails_before_stage_freeze(self) -> None:
        stage, result, preflight = self.make_stage_inputs()
        metadata_path = stage / "REPRODUCIBILITY_METADATA.json"
        metadata = json.loads(metadata_path.read_text())
        del metadata["hostname"]
        atomic_write_json(metadata_path, metadata)
        with self.assertRaisesRegex(PreparationError, "nonempty hostname"):
            prepare_frozen_stage(
                self.standalone,
                stage,
                result,
                preflight,
                result / "scheduler_selection.json",
                self.commit,
                self.tree,
                sys.executable,
                self.root,
            )


if __name__ == "__main__":
    unittest.main()
