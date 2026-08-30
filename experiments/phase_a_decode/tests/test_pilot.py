from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from experiments.phase_a_decode import pilot


def decision_rows(primary_speedup=1.30, short_h_over_j=1.0, *, short_stable=True,
                  primary_stable=True):
    return [
        {"S": 192, "Tkv": 4096, "stable": short_stable, "j_hot_p50_ms": 1.0,
         "h_hot_p50_ms": short_h_over_j, "jh_hot_speedup": 1.0 / short_h_over_j},
        {"S": 192, "Tkv": 32768, "stable": primary_stable, "j_hot_p50_ms": 1.0,
         "h_hot_p50_ms": 1.0 / primary_speedup, "jh_hot_speedup": primary_speedup},
    ]


def complete_measurement_row(t_kv, *, speedup=1.30, stable=True):
    row = {"S": 192, "Tkv": t_kv, "roles": 2, "kv_heads": 8, "head_dim": 128,
           "chunks": 1, "output_bytes": 8, "warmup_ms": 25, "rep_ms": 200,
           "outer_trials": 5, "j_config": "b256-w4", "f_config": "b256-w4",
           "h_config": "b256-w4", "jh_hot_speedup": speedup, "jh_cold_speedup": speedup,
           "stable": stable, "h_max_abs": 0.0, "h_relative_fro": 0.0}
    for variant in ("j", "f", "h"):
        for mode in ("cold", "hot"):
            row.update({f"{variant}_{mode}_p20_ms": 0.9,
                        f"{variant}_{mode}_p50_ms": 1.0,
                        f"{variant}_{mode}_p80_ms": 1.04,
                        f"{variant}_{mode}_gchunks_s": 1.0,
                        f"{variant}_{mode}_output_gib_s": 1.0,
                        f"{variant}_{mode}_stability": 0.04})
    row["j_hot_p50_ms"] = 1.0
    row["h_hot_p50_ms"] = 1.0 / speedup
    return row


def load_benchmark_artifact_writer():
    class Figure:
        def tight_layout(self):
            pass

        def savefig(self, path, dpi):
            Path(path).write_bytes(b"test-plot\n")

    class Axes:
        def plot(self, *args, **kwargs):
            pass

        def axhline(self, *args, **kwargs):
            pass

        def set(self, **kwargs):
            pass

        def set_xticks(self, *args):
            pass

        def grid(self, **kwargs):
            pass

        def legend(self, **kwargs):
            pass

    matplotlib = types.ModuleType("matplotlib")
    matplotlib.__path__ = []
    matplotlib.use = lambda *args, **kwargs: None
    pyplot = types.ModuleType("matplotlib.pyplot")
    pyplot.subplots = lambda **kwargs: (Figure(), Axes())
    pyplot.close = lambda figure: None
    matplotlib.pyplot = pyplot

    triton = types.ModuleType("triton")
    triton.__path__ = []
    triton.jit = lambda function: function
    language = types.ModuleType("triton.language")
    language.constexpr = object()
    triton.language = language

    modules = {"matplotlib": matplotlib, "matplotlib.pyplot": pyplot,
               "torch": types.ModuleType("torch"), "triton": triton,
               "triton.language": language}
    benchmark_path = Path(pilot.__file__).with_name("benchmark.py")
    spec = importlib.util.spec_from_file_location("_pilot_artifact_benchmark", benchmark_path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module.write_artifacts


class FakeCuda:
    def __init__(self):
        self.seed = None

    def is_available(self):
        return True

    def manual_seed_all(self, seed):
        self.seed = seed

    def get_device_name(self, index):
        return "fake-gpu"

    def get_device_capability(self, index):
        return (8, 6)

    def get_device_properties(self, index):
        return types.SimpleNamespace(total_memory=8 * 2**30)

    def mem_get_info(self, index=0):
        return 7 * 2**30, 8 * 2**30

    def device_count(self):
        return 1

    def is_bf16_supported(self):
        return True


class FakeTorch:
    __version__ = "test"
    version = types.SimpleNamespace(cuda="test")

    def __init__(self):
        self.cuda = FakeCuda()
        self.seed = None

    def manual_seed(self, seed):
        self.seed = seed


class FakeBenchmark:
    CONFIGS = ((256, 4, "b256-w4"), (512, 8, "b512-w8"))
    VARIANTS = ("J", "F", "H")

    def __init__(self, output, *, correctness_error=None):
        self.output = output
        self.correctness_error = correctness_error
        self.measure_calls = []
        self.tune_calls = 0
        self.write_calls = []

    def git(self, *args):
        return "fake"

    def run_correctness(self, args):
        if self.correctness_error is not None:
            raise self.correctness_error
        return {192: object()}, [{"dtype": "float16", "S": 192}], "passed"

    def tune(self, args, s_size, tables):
        self.tune_calls += 1
        selected = {variant: self.CONFIGS[0] for variant in self.VARIANTS}
        return selected, {variant: {"b256-w4": 1.0} for variant in self.VARIANTS}

    def correctness_summary(self, records, s_size):
        return {"h_max_abs_vs_j": 0.0, "h_max_abs_vs_oracle": 0.0,
                "h_relative_fro_vs_j": 0.0, "h_relative_fro_vs_oracle": 0.0}

    def measure(self, args, s_size, t_kv, tables, selected):
        self.measure_calls.append((s_size, t_kv))
        speedup = 1.0 if t_kv == 4096 else 1.30
        return complete_measurement_row(t_kv, speedup=speedup), [{"rep_ms": 200}]

    def write_artifacts(self, output, rows, raw, correctness, bf16, tuning, conclusion):
        self.write_calls.append({"rows": rows, "raw": raw, "correctness": correctness,
                                 "bf16": bf16, "tuning": tuning, "conclusion": conclusion})
        for name in pilot.SCIENTIFIC_ARTIFACTS:
            (output / name).write_text("test\n")


class PilotArgumentsTest(unittest.TestCase):
    def test_exact_arguments_pass_and_each_protocol_drift_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            args = pilot.build_parser().parse_args(["--output", str(output)])
            pilot.validate_args(args)
            changes = {"seed": 1, "s": [96], "t_kv": [32768, 4096], "roles": 1,
                       "kv_heads": 4, "head_dim": 64, "warmup": 10, "rep": 100,
                       "outer_trials": 3}
            for name, value in changes.items():
                with self.subTest(name=name):
                    changed = argparse.Namespace(**vars(args))
                    setattr(changed, name, value)
                    with self.assertRaisesRegex(ValueError, "do not match"):
                        pilot.validate_args(changed)

    def test_relative_output_is_rejected(self):
        args = pilot.build_parser().parse_args(["--output", "relative-result"])
        with self.assertRaisesRegex(ValueError, "absolute"):
            pilot.validate_args(args)


class PilotDecisionTest(unittest.TestCase):
    def test_go_optimize_and_kill_thresholds(self):
        cases = ((1.25, "GO"), (1.249, "OPTIMIZE ONCE"), (1.10, "OPTIMIZE ONCE"),
                 (1.099, "KILL"))
        for speedup, expected in cases:
            with self.subTest(speedup=speedup):
                self.assertEqual(pilot.decide_pilot(decision_rows(speedup)), expected)

    def test_short_regression_vetoes_primary_go(self):
        rows = decision_rows(1.40, short_h_over_j=1.050001)
        self.assertEqual(pilot.decide_pilot(rows), "KILL/RETARGET")
        self.assertIn("SHORT REGRESSION", rows[0]["decision"])

    def test_both_rows_must_be_stable(self):
        for short_stable, primary_stable in ((False, True), (True, False), (False, False)):
            with self.subTest(short=short_stable, primary=primary_stable):
                rows = decision_rows(1.40, short_stable=short_stable,
                                     primary_stable=primary_stable)
                self.assertEqual(pilot.decide_pilot(rows), "UNSTABLE — DO NOT INTERPRET")

    def test_mixed_stability_writes_complete_artifacts_without_interpretation(self):
        write_artifacts = load_benchmark_artifact_writer()
        unstable = "UNSTABLE — DO NOT INTERPRET"
        for short_stable, primary_stable in ((False, True), (True, False)):
            with self.subTest(short=short_stable, primary=primary_stable):
                rows = [complete_measurement_row(4096, speedup=1.0,
                                                 stable=short_stable),
                        complete_measurement_row(32768, speedup=1.40,
                                                 stable=primary_stable)]
                self.assertEqual(pilot.decide_pilot(rows), unstable)
                self.assertEqual([row["decision"] for row in rows],
                                 [unstable, unstable])
                with tempfile.TemporaryDirectory() as directory:
                    output = Path(directory)
                    write_artifacts(output, rows, {}, [], "passed", {}, unstable)
                    self.assertEqual({path.name for path in output.iterdir()},
                                     set(pilot.SCIENTIFIC_ARTIFACTS))
                    with (output / "timings.csv").open(newline="") as handle:
                        written = list(csv.DictReader(handle))
                    self.assertEqual([row["decision"] for row in written],
                                     [unstable, unstable])

    def test_exact_two_row_shape_is_required(self):
        with self.assertRaisesRegex(ValueError, "exactly"):
            pilot.decide_pilot(decision_rows()[:1])


class PilotControlFlowTest(unittest.TestCase):
    def runtime(self, benchmark):
        return pilot.Runtime(benchmark=benchmark, torch=FakeTorch(),
                             triton=types.SimpleNamespace(__version__="test"))

    def provenance(self, root):
        stage = root / "stage"
        source = root / "source"
        stage.mkdir()
        source.mkdir()
        metadata = stage / "REPRODUCIBILITY_METADATA.json"
        metadata.write_text("{}\n")
        return {"stage": stage, "source": source, "metadata": metadata}

    def test_correctness_failure_prevents_tuning_and_timing_and_is_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "result"
            args = pilot.build_parser().parse_args(["--output", str(output)])
            benchmark = FakeBenchmark(output, correctness_error=AssertionError("wrong"))
            with self.assertRaisesRegex(AssertionError, "wrong"):
                pilot.execute_pilot(args, self.runtime(benchmark), self.provenance(root), "pilot")
            self.assertEqual(benchmark.tune_calls, 0)
            self.assertEqual(benchmark.measure_calls, [])
            status = json.loads((output / "run_metadata.json").read_text())
            self.assertEqual((status["status"], status["phase"]), ("failed", "terminal"))

    def test_exact_two_rows_and_complete_artifact_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "result"
            args = pilot.build_parser().parse_args(["--output", str(output)])
            benchmark = FakeBenchmark(output)
            conclusion = pilot.execute_pilot(args, self.runtime(benchmark), self.provenance(root), "pilot")
            self.assertEqual(conclusion, "GO")
            self.assertEqual(benchmark.measure_calls, [(192, 4096), (192, 32768)])
            self.assertEqual(set(path.name for path in output.iterdir()),
                             set(pilot.ENVELOPE_ARTIFACTS + pilot.SCIENTIFIC_ARTIFACTS))
            write = benchmark.write_calls[0]
            self.assertEqual(list(write["raw"]), ["S192-T4096", "S192-T32768"])
            self.assertEqual(write["conclusion"], "GO")
            status = json.loads((output / "run_metadata.json").read_text())
            self.assertEqual((status["status"], status["conclusion"]), ("passed", "GO"))


class PilotAstTest(unittest.TestCase):
    def test_driver_adds_no_kernel_and_orders_correctness_before_timing(self):
        source = Path(pilot.__file__).read_text()
        tree = ast.parse(source)
        self.assertNotIn("triton.jit", source)
        execute = next(node for node in tree.body
                       if isinstance(node, ast.FunctionDef) and node.name == "execute_pilot")
        calls = [node for node in ast.walk(execute) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)]
        lines = {name: min(node.lineno for node in calls if node.func.attr == name)
                 for name in ("run_correctness", "tune", "measure")}
        self.assertLess(lines["run_correctness"], lines["tune"])
        self.assertLess(lines["tune"], lines["measure"])


if __name__ == "__main__":
    unittest.main()
