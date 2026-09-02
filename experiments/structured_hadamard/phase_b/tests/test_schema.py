from __future__ import annotations

import copy
import unittest

from experiments.structured_hadamard.phase_b import analysis, schema
from experiments.structured_hadamard.phase_b.reference import block_semantics


def _row(layer=0, transform="H32"):
    subset_quality = {
        name: {"rows": 4096, "normalized_output_error": 0.1,
               "squared_error_sum": 1.0, "reference_squared_sum": 10.0}
        for name in analysis.SUBSET_NAMES
    }
    subset_outlier = {
        name: {"rows": 4096, "max_abs": 2.0, "rms": 0.5}
        for name in analysis.SUBSET_NAMES
    }
    timing = {"minimum_ms": 0.1, "p10_ms": 0.1, "median_ms": 0.2,
              "p90_ms": 0.3, "maximum_ms": 0.4, "mean_ms": 0.2, "samples": 100}
    return {
        "schema_version": schema.ROW_SCHEMA_VERSION, "source_commit": "a" * 40,
        "layer": layer, "site": f"model.layers.{layer}.mlp.down_proj", "transform": transform,
        "model": {"repository": "NousResearch/Meta-Llama-3-8B", "revision": "r"},
        "data": {}, "quantization": {},
        "calibration": {"sequences": 16, "tokens_per_sequence": 512,
                        "sampled_token_rows": 8192, "dtype": "float16", "expanded": False},
        "shape": {"calibration_input": [8192, 14336], "timing_input": [1, 14336],
                  "reference_output": [8192, 4096]},
        "block_semantics": block_semantics(transform, 14336, full_outer_order=7),
        "quality": {"metric": "normalized_output_mse", "unit": "ratio",
                    "full": {"normalized_output_error": 0.1}, "subsets": subset_quality},
        "activation_outlier": {"full": {"max_abs": 2.0, "rms": 0.5},
                               "subsets": subset_outlier},
        "timing": {"unit": "milliseconds", "warmups": 20, "repetitions": 100,
                   "synchronization": "CUDA events with terminal torch.cuda.synchronize",
                   "transform": copy.deepcopy(timing),
                   "affected_packed_w4a4_layer": copy.deepcopy(timing)},
        "fusion": "none", "seed": 20260902,
        "hardware": {"name": "GPU", "uuid": "uuid", "compute_capability": [8, 6],
                     "total_memory_bytes": 1, "job_id": "1", "partition": "p",
                     "hostname": "node"},
    }


class PhaseBSchemaTest(unittest.TestCase):

    def test_complete_row(self):
        schema.validate_row(_row())

    def test_calibration_expansion_fails_closed(self):
        row = _row()
        row["calibration"]["expanded"] = True
        with self.assertRaises(schema.SchemaError):
            schema.validate_row(row)

    def test_exact_128_tuple_contract(self):
        rows = [_row(layer, transform) for layer in range(32)
                for transform in ("I", "H32", "H128", "Hfull")]
        schema.validate_rows(rows)
        with self.assertRaises(schema.SchemaError):
            schema.validate_rows(rows[:-1])


if __name__ == "__main__":
    unittest.main()
