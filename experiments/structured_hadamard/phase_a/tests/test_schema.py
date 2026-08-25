from __future__ import annotations

import copy
import json
import unittest

from experiments.structured_hadamard.phase_a.preflight import build_plan
from experiments.structured_hadamard.phase_a.schema import (BASE_COMMIT, ContractError, dumps_jsonl, loads_jsonl,
                                                            validate_record, validate_records)


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
        duplicated = '{"schema_version":"rot-site-v1.phase-a.1",' + encoded[1:]
        with self.assertRaisesRegex(ContractError, "duplicate JSON object key"):
            loads_jsonl(duplicated + "\n")


if __name__ == "__main__":
    unittest.main()
