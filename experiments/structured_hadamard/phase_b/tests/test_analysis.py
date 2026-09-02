from __future__ import annotations

import unittest

from experiments.structured_hadamard.phase_b import analysis


def _rows(*, heterogeneous=True, latency_gap=True):
    rows = []
    for layer in range(32):
        sensitivity = float(layer + 1) if heterogeneous else 1.0
        for transform in ("I", "H32", "H128", "Hfull"):
            error = sensitivity if transform == "H32" else 0.0
            layer_ms = 1.0 if transform == "H32" else (1.1 if transform == "Hfull" and latency_gap else 1.01)
            rows.append({
                "layer": layer, "site": f"model.layers.{layer}.mlp.down_proj", "transform": transform,
                "quality": {"subsets": {
                    analysis.SUBSET_NAMES[0]: {"normalized_output_error": error},
                    analysis.SUBSET_NAMES[1]: {"normalized_output_error": error * 2.0},
                }},
                "timing": {"affected_packed_w4a4_layer": {"median_ms": layer_ms}},
            })
    return rows


class PhaseBAnalysisTest(unittest.TestCase):

    def test_frozen_primary_ranking_and_stability_are_predeclared(self):
        frozen = analysis.freeze_selection(_rows())
        self.assertEqual(frozen["proxy"], analysis.PRIMARY_PROXY)
        self.assertEqual(frozen["cheaper_transform"], "H32")
        self.assertEqual(frozen["most_sensitive"], [31, 30, 29])
        self.assertEqual(frozen["least_sensitive"], [2, 1, 0])
        self.assertEqual(frozen["stability"]["spearman_rho"], 1.0)
        self.assertTrue(frozen["frozen_before_ppl"])

    def test_spearman_ties_and_reverse_order(self):
        self.assertAlmostEqual(
            analysis.spearman_rho({0: 1.0, 1: 2.0, 2: 3.0}, {0: 3.0, 1: 2.0, 2: 1.0}),
            -1.0,
        )
        self.assertEqual(
            analysis.spearman_rho({0: 1.0, 1: 1.0}, {0: 2.0, 1: 3.0}), 0.0,
        )

    def test_six_site_proxy_agreement_is_literal(self):
        frozen = analysis.freeze_selection(_rows())
        validations = [
            {"layer": layer, "ppl_impact_vs_phase_a_hfull": float(layer)}
            for layer in frozen["selected_sites"]
        ]
        agreement = analysis.proxy_agreement(frozen, validations)
        self.assertTrue(agreement["agrees"])
        self.assertGreater(agreement["six_site_spearman_rho"], 0.0)

    def test_literal_decision_b_branches(self):
        heterogeneous = _rows(heterogeneous=True, latency_gap=True)
        scores = analysis.primary_scores(heterogeneous)["mean"]
        b1 = analysis.literal_decision(
            heterogeneous, scores, stability_rho=1.0, proxy_validated=True,
        )
        self.assertTrue(b1["conclusion"].startswith("B1:"))
        b2_rows = _rows(heterogeneous=True, latency_gap=False)
        b2 = analysis.literal_decision(
            b2_rows, analysis.primary_scores(b2_rows)["mean"],
            stability_rho=1.0, proxy_validated=True,
        )
        self.assertTrue(b2["conclusion"].startswith("B2:"))
        uniform = _rows(heterogeneous=False, latency_gap=True)
        b3 = analysis.literal_decision(
            uniform, analysis.primary_scores(uniform)["mean"],
            stability_rho=1.0, proxy_validated=True,
        )
        self.assertTrue(b3["conclusion"].startswith("B3:"))
        b4_rows = _rows(heterogeneous=False, latency_gap=False)
        b4 = analysis.literal_decision(
            b4_rows, analysis.primary_scores(b4_rows)["mean"],
            stability_rho=1.0, proxy_validated=True,
        )
        self.assertTrue(b4["conclusion"].startswith("B4:"))


if __name__ == "__main__":
    unittest.main()
