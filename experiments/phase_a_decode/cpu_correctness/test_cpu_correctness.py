"""Deterministic, CPU-only tests for the Phase A J/F/H contract."""

from __future__ import annotations

import unittest

import torch

from experiments.phase_a_decode.cpu_correctness.oracle import (
    PRIMARY_COUNT,
    build_joint_table,
    decode_factorized,
    decode_joint,
    decode_specialized,
    decode_variants,
    hamilton,
    primary_units,
    split_ids,
)

SEEDS = (0, 20260824, 0xC0FFEE)


def random_unit_quaternions(size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.randn((size, 4), generator=generator, dtype=torch.float32)
    return values / torch.linalg.vector_norm(values, dim=-1, keepdim=True).clamp_min(1e-12)


class HurwitzContractTests(unittest.TestCase):
    def assert_variants_close(
        self,
        outputs: dict[str, torch.Tensor],
        *,
        atol: float = 0.0,
        rtol: float = 0.0,
    ) -> None:
        for variant in ("F", "H"):
            torch.testing.assert_close(outputs[variant], outputs["J"], atol=atol, rtol=rtol)

    def test_primary_catalog_exhausts_24_units_in_decoder_order(self) -> None:
        units = primary_units()
        self.assertEqual(units.shape, (24, 4))
        self.assertEqual(torch.unique(units, dim=0).shape[0], 24)
        torch.testing.assert_close(
            torch.linalg.vector_norm(units, dim=-1), torch.ones(24), atol=0, rtol=0
        )
        expected_axis = torch.tensor(
            [
                [1, 0, 0, 0],
                [-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, -1],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(units[:8], expected_axis, atol=0, rtol=0)
        for half_index in range(16):
            expected = torch.tensor(
                [
                    -0.5 if half_index & (1 << component) else 0.5
                    for component in range(4)
                ]
            )
            torch.testing.assert_close(units[8 + half_index], expected, atol=0, rtol=0)

    def test_scalar_first_hamilton_basis_orientation(self) -> None:
        one, i, j, k = torch.eye(4)
        cases = (
            (one, k, k),
            (i, j, k),
            (j, i, -k),
            (j, k, i),
            (k, j, -i),
            (-one, i, -i),
        )
        for left, right, expected in cases:
            with self.subTest(left=left.tolist(), right=right.tolist()):
                torch.testing.assert_close(hamilton(left, right), expected, atol=0, rtol=0)

    def test_flat_id_is_primary_times_S_plus_secondary(self) -> None:
        secondary_size = 7
        ids = torch.arange(PRIMARY_COUNT * secondary_size, dtype=torch.int32)
        p, s = split_ids(ids, secondary_size)
        torch.testing.assert_close(
            p, torch.arange(PRIMARY_COUNT).repeat_interleave(secondary_size), atol=0, rtol=0
        )
        torch.testing.assert_close(
            s, torch.arange(secondary_size).repeat(PRIMARY_COUNT), atol=0, rtol=0
        )
        torch.testing.assert_close(ids.to(torch.int64), p * secondary_size + s, atol=0, rtol=0)

    def test_axis_units_are_exact_signed_permutations(self) -> None:
        secondary = torch.tensor(
            [[2.0, -3.0, 5.0, -7.0], [-11.0, 13.0, -17.0, 19.0]]
        )
        expected_first = torch.tensor(
            [
                [2, -3, 5, -7],
                [-2, 3, -5, 7],
                [3, 2, 7, 5],
                [-3, -2, -7, -5],
                [-5, -7, 2, 3],
                [5, 7, -2, -3],
                [7, -5, -3, 2],
                [-7, 5, 3, -2],
            ],
            dtype=torch.float32,
        )
        ids = torch.arange(8, dtype=torch.int64) * secondary.shape[0]
        actual = decode_specialized(ids, secondary)
        torch.testing.assert_close(actual, expected_first, atol=0, rtol=0)
        torch.testing.assert_close(actual, decode_factorized(ids, secondary), atol=0, rtol=0)

    def test_all_half_units_match_generic_product_on_edge_and_sign_patterns(self) -> None:
        patterns = [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [2.0, -3.0, 5.0, -7.0],
            [2**-12, -(2**-8), 2**8, -(2**12)],
        ]
        patterns.extend(
            [
                [-1.0 if bits & (1 << component) else 1.0 for component in range(4)]
                for bits in range(16)
            ]
        )
        secondary = torch.tensor(patterns, dtype=torch.float32)
        secondary_size = secondary.shape[0]
        ids = torch.tensor(
            [p * secondary_size + s for p in range(8, 24) for s in range(secondary_size)]
        )
        specialized = decode_specialized(ids, secondary)
        generic = decode_factorized(ids, secondary)
        torch.testing.assert_close(specialized, generic, atol=0, rtol=0)

    def test_exhaustive_primary_and_secondary_indices_have_J_F_H_equality(self) -> None:
        for secondary_size in (1, 2, 7, 31):
            secondary = random_unit_quaternions(secondary_size, 1000 + secondary_size)
            ids = torch.arange(PRIMARY_COUNT * secondary_size, dtype=torch.int32)
            with self.subTest(secondary_size=secondary_size):
                outputs = decode_variants(ids, secondary)
                self.assert_variants_close(outputs, atol=2e-7, rtol=2e-7)
                axis = ids.to(torch.int64) // secondary_size < 8
                for variant in ("F", "H"):
                    self.assertTrue(torch.equal(outputs[variant][axis], outputs["J"][axis]))

    def test_joint_layout_gathers_every_flat_id(self) -> None:
        secondary = torch.tensor(
            [[1.0, 2.0, 4.0, 8.0], [-1.0, 3.0, -9.0, 27.0], [0.5, -0.25, 0.125, -0.0625]]
        )
        ids = torch.arange(PRIMARY_COUNT * secondary.shape[0], dtype=torch.int64)
        joint = build_joint_table(secondary)
        gathered = decode_joint(ids, joint, secondary.shape[0])
        generic = decode_factorized(ids, secondary)
        torch.testing.assert_close(gathered, generic, atol=0, rtol=0)

    def test_fp16_and_bf16_storage_round_trips(self) -> None:
        secondary = torch.cat(
            (
                random_unit_quaternions(37, 4242),
                torch.tensor(
                    [
                        [0.0, -0.0, 1.0, -1.0],
                        [2**-10, -(2**-7), 2**7, -(2**10)],
                        [-1.0, 1.0, -1.0, 1.0],
                    ]
                ),
            )
        )
        ids = torch.arange(PRIMARY_COUNT * secondary.shape[0], dtype=torch.int32)
        reference = decode_factorized(ids, secondary)
        tolerances = {
            torch.float16: (2e-3, 2e-3),
            torch.bfloat16: (2e-2, 2e-2),
        }
        for dtype, (atol, rtol) in tolerances.items():
            with self.subTest(dtype=str(dtype)):
                outputs = decode_variants(ids, secondary, dtype)
                self.assertEqual(outputs["J"].dtype, dtype)
                self.assert_variants_close(outputs, atol=atol, rtol=rtol)
                for variant in ("J", "F", "H"):
                    torch.testing.assert_close(
                        outputs[variant].to(torch.float32), reference, atol=atol, rtol=rtol
                    )

    def test_seeded_random_property_cases_are_reproducible(self) -> None:
        for seed in SEEDS:
            secondary_size = 3 + seed % 29
            secondary_a = random_unit_quaternions(secondary_size, seed)
            secondary_b = random_unit_quaternions(secondary_size, seed)
            self.assertTrue(torch.equal(secondary_a, secondary_b))
            generator_a = torch.Generator(device="cpu").manual_seed(seed ^ 0x5A5A)
            generator_b = torch.Generator(device="cpu").manual_seed(seed ^ 0x5A5A)
            ids_a = torch.randint(
                PRIMARY_COUNT * secondary_size, (4099,), generator=generator_a
            )
            ids_b = torch.randint(
                PRIMARY_COUNT * secondary_size, (4099,), generator=generator_b
            )
            self.assertTrue(torch.equal(ids_a, ids_b))
            with self.subTest(seed=seed, secondary_size=secondary_size):
                outputs_a = decode_variants(ids_a, secondary_a)
                outputs_b = decode_variants(ids_b, secondary_b)
                self.assert_variants_close(outputs_a, atol=2e-7, rtol=2e-7)
                for variant in ("J", "F", "H"):
                    self.assertTrue(torch.equal(outputs_a[variant], outputs_b[variant]))

    def test_invalid_ids_and_non_cpu_contract_fail_closed(self) -> None:
        secondary = torch.zeros((2, 4))
        for ids in (torch.tensor([-1]), torch.tensor([PRIMARY_COUNT * 2])):
            with self.subTest(ids=ids.tolist()), self.assertRaises(IndexError):
                decode_factorized(ids, secondary)
        with self.assertRaises(TypeError):
            split_ids(torch.tensor([0.0]), 2)
        with self.assertRaises(ValueError):
            split_ids(torch.tensor([0]), 0)
        with self.assertRaises(ValueError):
            decode_factorized(torch.tensor([0], device="meta"), secondary)


if __name__ == "__main__":
    unittest.main()
