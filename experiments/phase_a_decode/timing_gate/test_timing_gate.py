from __future__ import annotations

import json
import os

import pytest

from contract import SECONDARY_SIZES, benchmark_argv, classify, snapshot
from inventory import build
from runtime import atomic_json, finalize, payload_start, start, validate_timing


def rows(speedups=(1.0, 1.0), stable=True):
    output = []
    for size, speedup in zip(SECONDARY_SIZES, speedups, strict=True):
        variants = {
            variant: {
                "p20_us": 9.9,
                "p50_us": 10.0,
                "p80_us": 10.1,
                "dispersion": 0.02,
                "outer_stability": 0.02,
            }
            for variant in ("J", "F", "H")
        }
        output.append(
            {
                "S": size,
                "Tkv": 32768,
                "variants": variants,
                "jh_speedup": speedup,
                "stable": stable,
            }
        )
    return output


@pytest.mark.parametrize(
    ("speedups", "expected"),
    [
        ((1.09, 1.099), "KILL"),
        ((1.10, 1.09), "OPTIMIZE-ONCE"),
        ((1.249, 1.20), "OPTIMIZE-ONCE"),
        ((1.25, 1.00), "GO"),
    ],
)
def test_classification(speedups, expected):
    assert classify(rows(speedups)) == expected


def test_instability_and_shape_are_no_result():
    assert classify(rows(stable=False)) == "NO RESULT"
    assert classify(rows()[:1]) == "NO RESULT"


def test_contract_argv_is_exact():
    argv = benchmark_argv("/env/python", "/scratch/stage", "/research/result")
    assert argv[0] == "/env/python"
    assert argv[1] == "-B"
    assert "--correctness-only" not in argv
    assert argv[argv.index("--s") + 1 : argv.index("--t-kv")] == ["96", "192"]
    assert argv[argv.index("--t-kv") + 1] == "32768"
    contract = snapshot()
    assert contract["decoded_chunks_per_launch"] == 16_777_216
    assert contract["measurement"]["configuration_tuning"] is False
    assert contract["correctness_checkpoint"]["rerun_forbidden"] is True


def test_inventory_detects_hardlinks(tmp_path):
    original = tmp_path / "one"
    linked = tmp_path / "two"
    original.write_text("same")
    os.link(original, linked)
    with pytest.raises(ValueError, match="hard-linked"):
        build(tmp_path)


def test_inventory_changes_with_bytes(tmp_path):
    path = tmp_path / "value"
    path.write_text("one")
    before = build(tmp_path)["content_digest"]
    path.write_text("two")
    assert build(tmp_path)["content_digest"] != before


def test_runtime_start_accepts_only_own_slurm_log(tmp_path, monkeypatch):
    for name in (
        "attempt_ledger.jsonl",
        "command.txt",
        "environment.lock.txt",
        "environment_identity.json",
        "launch_manifest.json",
        "launch_manifest.sha256",
        "scheduler_options.json",
        "stage_inventory.json",
        "submission_request.json",
        "slurm-123.out",
    ):
        (tmp_path / name).write_text("")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    start(tmp_path, "c" * 64, "/research/source", "/scratch/stage")
    value = json.loads((tmp_path / "runtime_start.json").read_text())
    assert value["job_id"] == "123"
    assert value["launch_manifest_sha256"] == "c" * 64


def test_timing_validation_and_final_manifest(tmp_path, monkeypatch):
    digest = "a" * 64
    payload = {
        "schema": "vq-phase-a-timing-result/v1",
        "launch_manifest_sha256": digest,
        "classification": "GO",
        "contract": snapshot(),
        "rows": rows((1.25, 1.0)),
    }
    atomic_json(tmp_path / "timing_output.json", payload)
    classification, errors = validate_timing(tmp_path, digest)
    assert classification == "GO"
    assert errors == []
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    atomic_json(
        tmp_path / "environment_validation.json",
        {"status": "passed", "launch_manifest_sha256": digest},
    )
    payload_start(
        tmp_path, digest, "/research/source", "/scratch/stage", "batch-primary"
    )
    final = finalize(tmp_path, 0, digest, "/research/source", "/scratch/stage")
    assert final["classification"] == "GO"
    assert final["job_id"] == "123"
    assert final["payload_start_count"] == 1
    assert json.loads((tmp_path / "final_result_manifest.json").read_text()) == final


def test_declared_decision_drift_fails_closed(tmp_path):
    digest = "b" * 64
    payload = {
        "schema": "vq-phase-a-timing-result/v1",
        "launch_manifest_sha256": digest,
        "classification": "GO",
        "contract": snapshot(),
        "rows": rows((1.0, 1.0)),
    }
    atomic_json(tmp_path / "timing_output.json", payload)
    classification, errors = validate_timing(tmp_path, digest)
    assert classification == "NO RESULT"
    assert "declared classification differs" in errors[0]


def test_payload_start_is_manifest_bound_and_exclusive(tmp_path, monkeypatch):
    digest = "d" * 64
    monkeypatch.setenv("SLURM_JOB_ID", "456")
    atomic_json(
        tmp_path / "environment_validation.json",
        {"status": "passed", "launch_manifest_sha256": digest},
    )
    payload_start(
        tmp_path, digest, "/research/source", "/scratch/stage", "batch-primary"
    )
    latch = json.loads((tmp_path / "payload_start_latch.json").read_text())
    assert latch["launch_manifest_sha256"] == digest
    assert latch["payload_start_ordinal"] == 1
    with pytest.raises(FileExistsError):
        payload_start(
            tmp_path,
            digest,
            "/research/source",
            "/scratch/stage",
            "interactive-fallback",
        )
