from __future__ import annotations

import copy
import json
import unittest

from experiments.structured_hadamard.phase_a.preflight import build_plan
from experiments.structured_hadamard.phase_a.reference import transform_spec
from experiments.structured_hadamard.phase_a.schema import (BASE_COMMIT, ContractError, SCHEMA_VERSION, dumps_jsonl,
                                                            loads_jsonl, validate_record, validate_records)


class RecordContractTest(unittest.TestCase):

    def setUp(self):
        self.records = build_plan(head=BASE_COMMIT, dirty=True)

    def test_planned_matrix_round_trips(self):
        encoded = dumps_jsonl(self.records)
        decoded = loads_jsonl(encoded)
        self.assertEqual(decoded, self.records)

    def test_missing_required_field_is_rejected(self):
        record = copy.deepcopy(self.records[0])
        del record["metrics"]["local_nmse"]
        with self.assertRaisesRegex(ContractError, "missing"):
            validate_record(record)

    def test_nonfinite_metric_is_rejected(self):
        record = copy.deepcopy(self.records[0])
        record["metrics"]["local_nmse"] = float("nan")
        with self.assertRaisesRegex(ContractError, "non-finite"):
            validate_record(record)

    def test_inconsistent_hfull_factorization_is_rejected(self):
        record = copy.deepcopy(self.records[1])
        record["transform"]["q"] = 128
        with self.assertRaisesRegex(ContractError, "factorization"):
            validate_record(record)

    def test_mislabeled_identity_work_is_rejected(self):
        record = copy.deepcopy(self.records[0])
        record["transform"]["implementation"] = "triton-copy-kernel"
        with self.assertRaisesRegex(ContractError, "host-no-launch"):
            validate_record(record)

    def test_duplicate_logical_rows_are_rejected(self):
        with self.assertRaisesRegex(ContractError, "duplicate logical row"):
            validate_records([self.records[0], copy.deepcopy(self.records[0])])

    def test_duplicate_json_object_keys_are_rejected(self):
        encoded = json.dumps(self.records[0], allow_nan=False, separators=(",", ":"))
        duplicated = '{"schema_version":' + json.dumps(SCHEMA_VERSION) + ',' + encoded[1:]
        with self.assertRaisesRegex(ContractError, "duplicate JSON object key"):
            loads_jsonl(duplicated + "\n")

    def _measurement_record(self):
        record = copy.deepcopy(self.records[1])
        record["code"]["dirty"] = False
        record["model"]["revision"] = "a" * 40
        record["quant"].update({
            "w_group_size": "128",
            "a_group_size": "per-row",
            "scale_granularity": "per-group-W4;dynamic-per-row-A4",
            "clip": "none",
            "calibration_dataset": "wikitext-2-raw-v1@" + "b" * 40,
        })
        record["hardware"] = {
            "gpu": "observed-gpu",
            "compute_capability": "8.6",
            "driver": "observed-driver",
            "cuda": "observed-cuda",
            "torch": "observed-torch",
            "triton": "observed-triton",
            "clock_policy": "observed-unlocked",
        }
        record["metrics"].update({
            "correctness_passed": True,
            "inverse_rel_error": 0.0,
            "inverse_max_abs_error": 0.0,
            "local_equivalence_rel_error": 0.0,
            "local_equivalence_max_abs_error": 0.0,
            "local_nmse": 0.0,
            "nmse_epsilon": 1e-12,
            "activation_absmax": 1.0,
            "activation_rms": 0.5,
            "rotation_us_p10": 1.0,
            "rotation_us_median": 2.0,
            "rotation_us_p90": 3.0,
        })
        record["execution"].update({
            "status": "measurement",
            "scheduler_clearance": True,
            "scientific_evidence": False,
            "synthetic_input": True,
            "transform_launches": 2,
            "transform_copies": 0,
            "total_launches": 2,
        })
        return record

    def test_synthetic_workload_cannot_be_mislabeled_as_scientific_evidence(self):
        record = self._measurement_record()
        record["execution"]["synthetic_input"] = False
        record["execution"]["scientific_evidence"] = True
        with self.assertRaisesRegex(ContractError, "must agree with workload.input_source"):
            validate_record(record)

        record = self._measurement_record()
        record["execution"]["scientific_evidence"] = True
        with self.assertRaisesRegex(ContractError, "cannot be labeled scientific evidence"):
            validate_record(record)

    def test_measurement_requires_immutable_model_revision(self):
        record = self._measurement_record()
        record["model"]["revision"] = "main"
        with self.assertRaisesRegex(ContractError, "immutable 40-character model revision"):
            validate_record(record)

    def test_measurement_requires_resolved_quant_provenance(self):
        for key in ("w_group_size", "clip"):
            with self.subTest(key=key):
                record = self._measurement_record()
                record["quant"][key] = "UNRESOLVED-PHASE-A-OWNER"
                with self.assertRaisesRegex(ContractError, f"resolved quant.{key}"):
                    validate_record(record)

        record = self._measurement_record()
        record["quant"]["calibration_dataset"] = "wikitext-2-raw-v1@main"
        with self.assertRaisesRegex(ContractError, "calibration_dataset=name@40-character-commit"):
            validate_record(record)

    def test_resolved_measurement_provenance_validates(self):
        record = self._measurement_record()
        self.assertIs(validate_record(record), record)

    def test_phase_a_measurements_reject_reference_only_transforms(self):
        for transform_id in ("H32", "H128"):
            with self.subTest(transform_id=transform_id):
                record = self._measurement_record()
                spec = transform_spec(transform_id)
                record["transform"].update({
                    "id": spec.id,
                    "block_size": spec.block_size,
                    "K": spec.K,
                    "q": spec.q,
                    "normalization": spec.normalization,
                    "matrix_digest": spec.matrix_digest,
                    "implementation": spec.implementation,
                })
                with self.assertRaisesRegex(ContractError, "measurements accept only"):
                    validate_record(record)

    def test_phase_a_never_claims_quantization_fusion(self):
        record = self._measurement_record()
        record["transform"]["fusion"] = "quantize"
        with self.assertRaisesRegex(ContractError, "fusion must be 'none'"):
            validate_record(record)

        record = self._measurement_record()
        record["timing"]["identity"] = "transform+quantize"
        record["timing"]["composition"] = "fused-transform-quantize"
        with self.assertRaisesRegex(ContractError, "sequential-transform-then-quantize"):
            validate_record(record)


if __name__ == "__main__":
    unittest.main()
