from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.structured_hadamard.phase_c import analysis, driver, gpu_owner, local_event_owner, schema, selector


ROOT = Path(__file__).resolve().parents[4]
MAP = ROOT / selector.PHASE_B_MAP_REPO_PATH
REPORT = ROOT / selector.PHASE_B_REPORT_REPO_PATH


def _synthetic_source(*, zero_saving: bool = False):
    source = {}
    costs = {"I": 0.0, "H32": 1.0, "H128": 2.0, "Hfull": 3.0}
    errors = {"I": 3.0, "H32": 2.0, "H128": 1.0, "Hfull": 0.0}
    if zero_saving:
        costs["H128"] = costs["Hfull"]
    for layer in range(32):
        for line, transform in enumerate(selector.TRANSFORM_ORDER, start=1):
            source[layer, transform] = {
                "line": layer * 4 + line,
                "sha256": f"{layer * 4 + line:064x}",
                "row": {
                    "timing": {"transform": {"median_ms": costs[transform]}},
                    "quality": {"full": {"normalized_output_error": errors[transform]}},
                },
            }
    return source


def _measured(freeze, values):
    rows = []
    for measurement_id in freeze["measurement_order"]:
        aliases = freeze["measurements"][measurement_id]["policy_aliases"]
        policy_id = aliases[0]
        ppl, milliseconds = values.get(policy_id, (100.0, 1000.0))
        rows.append({
            "measurement_id": measurement_id,
            "quality": {"perplexity": ppl},
            "decode": {
                "median_ms": milliseconds,
                "median_ms_per_output_token": milliseconds / 32.0,
                "median_tokens_per_second": 32000.0 / milliseconds,
            },
            "realized_predicted_transform_cost_ms":
                freeze["measurements"][measurement_id]["realized_predicted_transform_cost_ms"],
            "realized_predicted_transform_cost_percent":
                freeze["measurements"][measurement_id]["realized_predicted_transform_cost_percent"],
        })
    return rows


class PhaseCSelectorTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.freeze = selector.freeze_policies(MAP, REPORT)

    def test_real_phase_b_freeze_has_five_monotone_budgets_and_deduplicated_endpoints(self):
        value = self.freeze
        self.assertEqual([row["target_budget_percent"] for row in value["adaptive_policies"]],
                         [0, 25, 50, 75, 100])
        self.assertEqual(value["selector"]["trace_steps"], 96)
        self.assertEqual(value["deduplication"]["identity_count"], 9)
        self.assertEqual(value["deduplication"]["unique_measurement_count"], 7)
        mapping = value["identity_to_measurement"]
        self.assertEqual(mapping["adaptive-budget-000"], mapping["fixed-i"])
        self.assertEqual(mapping["adaptive-budget-100"], mapping["fixed-hfull"])
        ranks = {name: index for index, name in enumerate(selector.TRANSFORM_ORDER)}
        previous = None
        for policy in value["adaptive_policies"]:
            self.assertLessEqual(policy["realized_predicted_transform_cost_ms"],
                                 policy["target_predicted_transform_cost_ms"] + 1e-12)
            self.assertEqual(len(policy["source_rows"]), 32)
            if previous:
                self.assertTrue(all(ranks[previous[str(layer)]] <= ranks[policy["assignments"][str(layer)]]
                                    for layer in range(32)))
            previous = policy["assignments"]
        selector.validate_freeze(value, map_path=MAP, report_path=REPORT)

    def test_lowest_rho_tie_break_is_stable_and_layer_first(self):
        _full_cost, trace = selector._downgrade_trace(_synthetic_source())
        self.assertEqual(
            [(row["move"]["layer"], row["move"]["from"], row["move"]["to"])
             for row in trace[1:4]],
            [(0, "Hfull", "H128"), (0, "H128", "H32"), (0, "H32", "I")],
        )

    def test_nonpositive_adjacent_latency_saving_fails_closed(self):
        with self.assertRaisesRegex(selector.SelectorError, "strictly positive latency saving"):
            selector._downgrade_trace(_synthetic_source(zero_saving=True))

    def test_freeze_rejects_source_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "map.jsonl"
            changed.write_bytes(MAP.read_bytes() + b"\n")
            with self.assertRaisesRegex(selector.SelectorError, "digest differs"):
                selector.freeze_policies(changed, REPORT)


class PhaseCDecisionTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.freeze = selector.freeze_policies(MAP, REPORT)

    def test_literal_c1_c2_c3_c4_branches(self):
        common = {
            "adaptive-budget-000": (1000.0, 800.0),
            "adaptive-budget-025": (106.0, 850.0),
            "adaptive-budget-050": (106.0, 900.0),
            "fixed-i": (1000.0, 800.0),
            "fixed-h32": (105.0, 920.0),
            "fixed-h128": (101.0, 990.0),
            "adaptive-budget-100": (100.0, 1000.0),
            "fixed-hfull": (100.0, 1000.0),
        }
        cases = {
            "C1": {**common, "adaptive-budget-075": (100.1, 950.0)},
            "C2": {**common, "adaptive-budget-075": (100.1, 980.0)},
            "C3": {**common, "adaptive-budget-075": (101.0, 900.0),
                   "fixed-h128": (102.0, 990.0)},
            "C4": {**common, "adaptive-budget-025": (106.0, 950.0),
                   "adaptive-budget-050": (104.0, 995.0),
                   "adaptive-budget-075": (101.2, 1005.0)},
        }
        for expected, values in cases.items():
            with self.subTest(expected=expected):
                result = analysis.literal_decision(_measured(self.freeze, values), self.freeze)
                self.assertEqual(result["decision"], expected)
                self.assertFalse(result["phase_d_entered"])
                self.assertFalse(result["second_scientific_run_entered"])


class PhaseCContractTest(unittest.TestCase):

    def test_driver_defaults_preserve_accepted_phase_a_settings(self):
        args = driver._parser().parse_args([
            "--stage-root", "/stage", "--source-commit", "a" * 40,
            "--snapshot", "/snapshot", "--evaluation-arrow", "/test.arrow",
            "--extension", "/runtime.so", "--quarot-root", "/quarot",
            "--phase-a-results", "/phase-a.json", "--phase-b-map", "/map.jsonl",
            "--phase-b-report", "/report.md", "--policy-freeze", "/freeze.json",
            "--policy-freeze-sha256", "b" * 64, "--output-directory", "/output",
            "--status-path", "/status.jsonl",
        ])
        self.assertEqual((args.ppl_sequence_length, args.prompt_length, args.output_length),
                         (1024, 128, 32))
        self.assertEqual((args.decode_warmups, args.decode_repetitions), (1, 5))
        self.assertEqual(args.fusion, "none")

    def test_measured_schema_checks_real_runtime_contract(self):
        freeze = selector.freeze_policies(MAP, REPORT)
        measurement_id = freeze["measurement_order"][0]
        measurement = freeze["measurements"][measurement_id]
        row = {
            "schema_version": schema.ROW_SCHEMA_VERSION,
            "measurement_id": measurement_id, "policy_aliases": measurement["policy_aliases"],
            "assignment_sha256": measurement["assignment_sha256"],
            "assignments": measurement["assignments"], "source_rows": measurement["source_rows"],
            "source_commit": "a" * 40, "model": {}, "data": {}, "quantization": {},
            "quality": {"dataset": "Salesforce/wikitext", "sequence_length": 1024,
                        "perplexity": 100.0},
            "decode": {"prompt_length": 128, "output_length": 32, "batch_size": 1,
                       "warmups": 1, "repetitions": 5, "median_ms": 1000.0,
                       "median_ms_per_output_token": 31.25,
                       "median_tokens_per_second": 32.0},
            "realized_predicted_transform_cost_ms":
                measurement["realized_predicted_transform_cost_ms"],
            "realized_predicted_transform_cost_percent":
                measurement["realized_predicted_transform_cost_percent"],
            "predicted_aggregate_local_nmse_loss_vs_hfull": 1.0,
            "fusion": "none", "seed": 20260902, "hardware": {},
        }
        row["data"] = {"dataset": "Salesforce/wikitext"}
        schema.validate_record(row)
        changed = copy.deepcopy(row)
        changed["fusion"] = "fused"
        with self.assertRaises(schema.SchemaError):
            schema.validate_record(changed)

    def test_one_shot_owner_and_single_ssh_route_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.json"
            path.write_text(json.dumps({"owner_attempts": 0}), encoding="utf-8")
            self.assertEqual(gpu_owner._advance(path, "owner_attempts")["owner_attempts"], 1)
            with self.assertRaisesRegex(RuntimeError, "one-shot ledger refusal"):
                gpu_owner._advance(path, "owner_attempts")
        args = argparse.Namespace(
            ssh="/usr/bin/ssh", control_socket=Path("/tmp/phasec/ssh-control"),
            host="slurm-login.csail.mit.edu", remote_argv=["/pinned/python", "owner.py"],
        )
        self.assertEqual(local_event_owner.build_ssh_argv(args), [
            "/usr/bin/ssh", "-S", "/tmp/phasec/ssh-control", "-o", "ControlMaster=no",
            "-o", "BatchMode=yes", "-tt", "slurm-login.csail.mit.edu",
            "/pinned/python", "owner.py",
        ])
        line = local_event_owner._summary(
            "phasec-selector-r1", {"event": "allocation_acquired", "job_id": "42", "partition": "p"},
        )
        self.assertIn("[key=phasec-selector-r1]", line)
        self.assertIn("job=42", line)


if __name__ == "__main__":
    unittest.main()
