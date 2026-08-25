"""Unit tests for Phase A validation and descriptive aggregation."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from experiments.phase_a_decode.analysis.analyze import (
    SYNTHETIC_LABEL,
    aggregate,
    write_artifacts,
)
from experiments.phase_a_decode.analysis.make_synthetic_fixture import build_fixture
from experiments.phase_a_decode.analysis.schema import ValidationError, validate_result_directory


class AnalysisValidationTests(unittest.TestCase):

    def make_fixture(self, unstable_cell=None) -> tuple[tempfile.TemporaryDirectory, Path]:
        temporary = tempfile.TemporaryDirectory()
        path = Path(temporary.name) / "conspicuously-synthetic-inputs"
        build_fixture(path, unstable_cell=unstable_cell)
        return temporary, path

    def test_complete_fixture_validates_and_artifacts_are_watermarked(self):
        temporary, path = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        validated = validate_result_directory(path)
        self.assertEqual(len(validated["timings"]), 6)
        output = Path(temporary.name) / "generated"
        write_artifacts(path, output)
        csv_text = (output / "analysis_table.csv").read_text(encoding="utf-8")
        markdown = (output / "analysis_table.md").read_text(encoding="utf-8")
        svg = (output / "comparison.svg").read_text(encoding="utf-8")
        self.assertEqual(csv_text.count(SYNTHETIC_LABEL), 6)
        self.assertIn(f"# {SYNTHETIC_LABEL}", markdown)
        self.assertGreaterEqual(svg.count(SYNTHETIC_LABEL), 2)
        report = json.loads((output / "validation_report.json").read_text())
        self.assertIsNone(report["scientific_decision"])

    def test_aggregation_outputs_are_byte_deterministic(self):
        temporary, path = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        first = Path(temporary.name) / "first"
        second = Path(temporary.name) / "second"
        write_artifacts(path, first)
        write_artifacts(path, second)
        for name in ("analysis_table.csv", "analysis_table.md", "comparison.svg",
                     "validation_report.json"):
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

    def test_missing_input_fails_closed(self):
        temporary, path = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        (path / "correctness.json").unlink()
        with self.assertRaisesRegex(ValidationError, "missing required artifacts"):
            validate_result_directory(path)

    def test_malformed_input_fails_closed(self):
        temporary, path = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        (path / "tuning.json").write_text("{not-json", encoding="utf-8")
        with self.assertRaisesRegex(ValidationError, "malformed or unreadable JSON"):
            validate_result_directory(path)

    def test_duplicate_timing_cell_fails_closed(self):
        temporary, path = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        timing_path = path / "timings.csv"
        with timing_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        with timing_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerows(rows + [rows[1]])
        with self.assertRaisesRegex(ValidationError, "duplicate cell"):
            validate_result_directory(path)

    def test_duplicate_json_key_fails_closed(self):
        temporary, path = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        (path / "tuning.json").write_text(
            '{"configs": [], "configs": [], "selection_shape_tkv": 4096, "by_s": {}}',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValidationError, "duplicate object key"):
            validate_result_directory(path)

    def test_unstable_input_is_flagged_without_a_gate_decision(self):
        temporary, path = self.make_fixture(unstable_cell=(192, 32768))
        self.addCleanup(temporary.cleanup)
        rows = aggregate(validate_result_directory(path))
        target = next(row for row in rows if (row["S"], row["Tkv"]) == (192, 32768))
        self.assertEqual(target["stability_summary"], "UNSTABLE — DESCRIPTIVE ONLY")
        self.assertGreater(target["max_hot_stability"], 0.05)
        self.assertNotIn("decision", target)

    def test_partial_trial_matrix_fails_closed(self):
        temporary, path = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        trial_path = path / "trial_timings.json"
        data = json.loads(trial_path.read_text(encoding="utf-8"))
        data.pop("S192-T32768")
        trial_path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValidationError, "partial matrix"):
            validate_result_directory(path)

    def test_cross_file_quantile_mismatch_fails_closed(self):
        temporary, path = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        timing_path = path / "timings.csv"
        with timing_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fields = reader.fieldnames
        rows[0]["j_hot_p50_ms"] = str(float(rows[0]["j_hot_p50_ms"]) * 1.01)
        with timing_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaises(ValidationError):
            validate_result_directory(path)


if __name__ == "__main__":
    unittest.main()
