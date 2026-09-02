from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from experiments.structured_hadamard.phase_a.real_model import build_w4a4_extension
from experiments.structured_hadamard.phase_a.real_model.prepare import build_assembly
from experiments.structured_hadamard.phase_a.real_model import runtime


class _Tensor:

    def __init__(self, shape, label="tensor", dtype="float16", is_meta=False):
        self.shape = tuple(shape)
        self.label = label
        self.dtype = dtype
        self.is_meta = is_meta

    def detach(self):
        return _Tensor(self.shape, f"detach({self.label})", self.dtype)

    def float(self):
        return _Tensor(self.shape, f"float({self.label})", "float32")

    def abs(self):
        return _Tensor(self.shape, f"abs({self.label})", self.dtype)

    def amax(self, dim):
        shape = self.shape[:dim] + self.shape[dim + 1:]
        return _Tensor(shape, f"amax({self.label})", self.dtype)

    def clamp_min(self, value):
        return _Tensor(self.shape, f"clamp({self.label})", self.dtype)

    def to(self, *args, **kwargs):
        dtype = kwargs.get("dtype") or (args[0] if args else self.dtype)
        return _Tensor(self.shape, f"to({self.label})", dtype)

    def contiguous(self):
        return _Tensor(self.shape, f"contiguous({self.label})", self.dtype)

    def reshape(self, *shape):
        shape = tuple(shape)
        if -1 in shape:
            known = math.prod(value for value in shape if value != -1)
            shape = tuple(math.prod(self.shape) // known if value == -1 else value for value in shape)
        return _Tensor(shape, f"reshape({self.label})", self.dtype)

    def __truediv__(self, value):
        return _Tensor(self.shape, f"div({self.label})", self.dtype)


class _Module:

    def register_buffer(self, name, value):
        setattr(self, name, value)


class _Torch:
    nn = SimpleNamespace(Module=_Module)


class _Extension:

    def __init__(self, calls):
        self.calls = calls

    def sym_quant(self, tensor, scales):
        self.calls.append(("sym_quant", tensor.label, scales.label))
        return _Tensor((tensor.shape[0], tensor.shape[1] // 2), "packed", "uint8")

    def matmul(self, activation, weight):
        self.calls.append(("matmul", activation.label, weight.label))
        return _Tensor((activation.shape[0], weight.shape[0]), "accumulated", "int32")

    def sym_dequant(self, accumulated, row_scales, column_scales):
        self.calls.append(("sym_dequant", accumulated.label, row_scales.label, column_scales.label))
        return _Tensor(accumulated.shape, "dequantized", "float16")


class _Transform:

    def __init__(self, calls):
        self.calls = calls

    def fold_weight(self, tensor):
        self.calls.append(("fold_weight", tensor.label))
        return _Tensor(tensor.shape, "folded_weight", tensor.dtype)

    def online(self, tensor):
        self.calls.append(("online", tensor.label))
        return _Tensor(tensor.shape, "transformed_activation", tensor.dtype)


class RealModelPreparationTest(unittest.TestCase):

    def _config(self):
        return SimpleNamespace(
            hidden_size=4096,
            intermediate_size=14336,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
        )

    def _snapshot(self, root: Path):
        snapshot = root / runtime.MODEL_REVISION
        snapshot.mkdir()
        config = dict(vars(self._config()))
        (snapshot / "config.json").write_text(json.dumps(config), encoding="utf-8")
        weight_map = {"model.embed_tokens.weight": "one.safetensors", "lm_head.weight": "two.safetensors"}
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}), encoding="utf-8"
        )
        (snapshot / "one.safetensors").write_bytes(b"one")
        (snapshot / "two.safetensors").write_bytes(b"twotwo")
        metadata = {
            name: hashlib.sha256((snapshot / name).read_bytes()).hexdigest()
            for name in ("config.json", "model.safetensors.index.json")
        }
        blobs = {
            "one.safetensors": (3, hashlib.sha256(b"one").hexdigest()),
            "two.safetensors": (6, hashlib.sha256(b"twotwo").hexdigest()),
        }
        return snapshot, metadata, blobs

    def test_llama3_gqa_head_dimension_and_cache_shape(self):
        dimensions = runtime.verify_llama3_dimensions(self._config())
        self.assertEqual(dimensions.head_dim, 128)
        self.assertNotEqual(dimensions.head_dim, 4096 // 8)
        self.assertEqual(dimensions.cache_shape(2, 17), (2, 8, 17, 128))

    def test_exact_cached_weights_use_standard_model_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot, metadata, blobs = self._snapshot(Path(directory))
            state = {
                "model.embed_tokens.weight": _Tensor((2, 2), "embed"),
                "lm_head.weight": _Tensor((2, 2), "head"),
            }
            model = SimpleNamespace(config=self._config(), state_dict=lambda: state)
            loader = mock.Mock(return_value=model)
            transformers = SimpleNamespace(LlamaForCausalLM=SimpleNamespace(from_pretrained=loader))
            with mock.patch.object(runtime, "MODEL_METADATA_SHA256", metadata), \
                    mock.patch.object(runtime, "MODEL_WEIGHT_BLOBS", blobs):
                loaded = runtime.load_cached_llama3(snapshot, transformers_module=transformers)
            self.assertIs(loaded, model)
            loader.assert_called_once_with(
                str(snapshot), local_files_only=True, trust_remote_code=False, torch_dtype="float16",
                low_cpu_mem_usage=True, attn_implementation="eager",
            )

    def test_loaded_weight_verification_rejects_meta_or_wrong_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot, _, _ = self._snapshot(Path(directory))
            wrong = SimpleNamespace(state_dict=lambda: {"wrong": _Tensor((1,), "wrong")})
            with self.assertRaisesRegex(runtime.PreparationError, "state keys differ"):
                runtime.verify_loaded_weights(wrong, snapshot)
            meta = SimpleNamespace(state_dict=lambda: {
                "model.embed_tokens.weight": _Tensor((1,), "embed", is_meta=True),
                "lm_head.weight": _Tensor((1,), "head"),
            })
            with self.assertRaisesRegex(runtime.PreparationError, "uninitialized meta"):
                runtime.verify_loaded_weights(meta, snapshot)

    def test_only_selected_down_projection_is_replaced_and_routed(self):
        calls = []
        extension = _Extension(calls)
        original_zero = SimpleNamespace(
            in_features=4, out_features=2, bias=None, weight=_Tensor((2, 4), "weight0")
        )
        original_one = SimpleNamespace(
            in_features=4, out_features=2, bias=None, weight=_Tensor((2, 4), "weight1")
        )
        attention = (object(), object())
        layers = [
            SimpleNamespace(self_attn=attention[0], mlp=SimpleNamespace(down_proj=original_zero)),
            SimpleNamespace(self_attn=attention[1], mlp=SimpleNamespace(down_proj=original_one)),
        ]
        model = SimpleNamespace(model=SimpleNamespace(layers=layers))
        replaced = runtime.replace_down_projections(
            model, layers=(0,), extension=extension,
            transform_factory=lambda width: _Transform(calls), torch_module=_Torch,
        )
        self.assertEqual(replaced, (0,))
        self.assertIs(layers[0].self_attn, attention[0])
        self.assertIs(layers[1].self_attn, attention[1])
        self.assertIs(layers[1].mlp.down_proj, original_one)
        output = layers[0].mlp.down_proj.forward(_Tensor((1, 3, 4), "activation"))
        self.assertEqual(output.shape, (1, 3, 2))
        self.assertEqual([call[0] for call in calls], [
            "fold_weight", "sym_quant", "online", "sym_quant", "matmul", "sym_dequant",
        ])

    def test_all_projection_linears_are_packed_and_only_down_projection_rotates(self):
        def linear(in_features, out_features):
            return SimpleNamespace(
                in_features=in_features, out_features=out_features, bias=None,
                weight=_Tensor((out_features, in_features), "weight"),
            )

        def model():
            layers = []
            for _ in range(2):
                layers.append(SimpleNamespace(
                    self_attn=SimpleNamespace(
                        q_proj=linear(4, 4), k_proj=linear(4, 2),
                        v_proj=linear(4, 2), o_proj=linear(4, 4),
                    ),
                    mlp=SimpleNamespace(
                        gate_proj=linear(4, 8), up_proj=linear(4, 8), down_proj=linear(8, 4),
                    ),
                ))
            return SimpleNamespace(model=SimpleNamespace(layers=layers))

        calls = []
        identity_model = model()
        replaced = runtime.replace_all_projection_linears(
            identity_model, extension=_Extension(calls), rotate_down_projections=False,
            transform_factory=lambda _width: self.fail("identity path requested a transform"),
            torch_module=_Torch,
        )
        self.assertEqual(len(replaced), 14)
        self.assertTrue(all(isinstance(layer.mlp.down_proj.transform, runtime.IdentityTransform)
                            for layer in identity_model.model.layers))

        calls = []
        rotated_model = model()
        replaced = runtime.replace_all_projection_linears(
            rotated_model, extension=_Extension(calls), rotate_down_projections=True,
            transform_factory=lambda _width: _Transform(calls), torch_module=_Torch,
        )
        self.assertEqual(len(replaced), 14)
        self.assertTrue(all(isinstance(layer.mlp.down_proj.transform, _Transform)
                            for layer in rotated_model.model.layers))
        self.assertTrue(all(isinstance(layer.self_attn.q_proj.transform, runtime.IdentityTransform)
                            for layer in rotated_model.model.layers))

    def test_transform_inverse_and_fold_orientation(self):
        result = runtime.unquantized_one_block_smoke()
        self.assertLessEqual(result["inverse_max_abs"], 1.0e-12)
        self.assertLessEqual(result["equivalence_max_abs"], 1.0e-12)

    def test_signed_packing_scales_and_int32_dequant_reference(self):
        rows = [[-8.0, -7.0, -0.5, 0.5, 6.9, 7.0, 0.0, 1.0]]
        scales = runtime.signed_int4_scale_rows(rows)
        quantized = runtime.quantize_signed_int4_rows(rows, scales)
        packed = runtime.pack_signed_int4_rows(quantized)
        self.assertEqual(runtime.unpack_signed_int4_rows(packed, rows=1, columns=8), quantized)
        self.assertTrue(all(runtime.SIGNED_INT4_MIN <= value <= runtime.SIGNED_INT4_MAX
                            for value in quantized[0]))
        activation_q = [[-8, 7, 2, -1]]
        weight_q = [[1, -2, 3, 4], [-1, 1, -1, 1]]
        accumulated = [sum(a * w for a, w in zip(activation_q[0], row)) for row in weight_q]
        row_scale, column_scales = 0.25, [0.5, 2.0]
        self.assertEqual(
            [value * row_scale * column_scale for value, column_scale in zip(accumulated, column_scales)],
            [-2.5, 6.0],
        )

    def test_cli_assembly_and_verified_architecture_boundary(self):
        self.assertEqual(build_w4a4_extension.SASS_TARGETS, ("sm_80", "sm_86"))
        self.assertEqual(build_w4a4_extension.PTX_TARGETS, ("compute_80",))
        self.assertEqual(build_w4a4_extension.TORCH_CUDA_ARCH_LIST, "8.0+PTX;8.6")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot, metadata, blobs = self._snapshot(root)
            extension = root / "phase_a_w4a4_cuda.so"
            extension.write_bytes(b"extension")
            quarot = root / "quarot"
            quarot.mkdir()
            args = argparse.Namespace(
                snapshot=snapshot,
                extension=extension,
                extension_sha256=hashlib.sha256(b"extension").hexdigest(),
                quarot_root=quarot,
                layer=[],
                device="cuda:0",
                hash_weight_blobs=True,
                load_model=False,
            )
            with mock.patch.object(runtime, "MODEL_METADATA_SHA256", metadata), \
                    mock.patch.object(runtime, "MODEL_WEIGHT_BLOBS", blobs):
                assembly = build_assembly(args)
        self.assertEqual(assembly["layers"], [0])
        self.assertEqual(assembly["replacement_site"], "model.layers[*].mlp.down_proj")
        self.assertEqual(assembly["attention"], "transformers-standard-eager")
        self.assertFalse(assembly["model_loaded"])
        self.assertFalse(assembly["inference_run"])


if __name__ == "__main__":
    unittest.main()
