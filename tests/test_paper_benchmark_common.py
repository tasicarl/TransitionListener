"""Regression tests for the shared paper-benchmark helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "arxiv" / "reproducibility" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _paper_benchmark_common import (  # noqa: E402
    BENCHMARK_CONFIG,
    cache_matches,
    cache_metadata,
    load_benchmark_spec,
    temperature_axis_scale,
)
from transitionlistener.phases import Phases  # noqa: E402


class PaperBenchmarkCommonTests(unittest.TestCase):
    def test_load_benchmark_spec_reads_shared_flipflop_point(self):
        spec = load_benchmark_spec()

        self.assertEqual(spec.config_path, BENCHMARK_CONFIG.resolve())
        self.assertEqual(spec.potential_name, "DarkFlipFlop")
        self.assertTrue(str(spec.modelfile).endswith("models/TL_dark_flipflop.py"))
        self.assertAlmostEqual(float(spec.params["lambda0"]), 0.005098)
        self.assertAlmostEqual(float(spec.params["gamma"]), 0.7532)
        self.assertEqual(spec.additional_plots["profileV"]["field_index_1"], 0)
        self.assertEqual(spec.additional_plots["action"]["phase_indices"], ["P1", "P3"])

    def test_cache_matches_requires_matching_version_and_signature(self):
        spec = load_benchmark_spec()
        payload = {"cache_meta": cache_metadata(spec, 3)}

        self.assertTrue(cache_matches(payload, spec, 3))
        self.assertFalse(cache_matches(payload, spec, 4))
        self.assertFalse(cache_matches({}, spec, 3))

    def test_temperature_axis_scale_switches_between_gev_and_mev(self):
        scale_small, label_small = temperature_axis_scale([0.02, 0.03])
        scale_large, label_large = temperature_axis_scale([0.5, 1.2])

        self.assertEqual(scale_small, 1.0e3)
        self.assertIn("MeV", label_small)
        self.assertEqual(scale_large, 1.0)
        self.assertIn("GeV", label_large)

    def test_phase_aliases_resolve_to_internal_keys(self):
        phases = Phases.__new__(Phases)
        phases.phases = {0: None, 1: None, "2-5-2-5": None}

        self.assertEqual(phases.phase_alias(0), "P1")
        self.assertEqual(phases.phase_alias("2-5-2-5"), "P3")
        self.assertEqual(phases.resolve_phase_key("P1"), 0)
        self.assertEqual(phases.resolve_phase_key("P3"), "2-5-2-5")


if __name__ == "__main__":
    unittest.main()
