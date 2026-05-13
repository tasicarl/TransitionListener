"""Regression tests for extended warning observables and output plumbing."""

from __future__ import annotations

import types
import unittest

import numpy as np

from transitionlistener import config
from transitionlistener.interface.output_schema import WARNING_COLUMN_ORDER
from transitionlistener.interface.samplers import get_empty_result
from transitionlistener.transitionObservables import TransitionObservables


class WarningOutputTests(unittest.TestCase):
    def test_warning_observables_are_exposed_in_default_metadata(self):
        for key in (
            "WARNING:betaH_small",
            "WARNING:betaH_very_small",
            "WARNING:betaH_mismatch",
            "WARNING:betaH_nonfinite",
            "WARNING:nucleationRate_nonexponential",
        ):
            with self.subTest(key=key):
                self.assertIn(key, config.all_observables)
                self.assertIn(key, WARNING_COLUMN_ORDER)
                self.assertIn(key, get_empty_result())

    def test_finalize_results_restores_warning_semantics(self):
        obs = TransitionObservables.__new__(TransitionObservables)

        ctx = types.SimpleNamespace(
            derived_params={
                "betaH_S3": 120.0,
                "betaH_RH": 2.5,
            },
            derived_param_names=[
                "WARNING:betaH_small",
                "WARNING:betaH_very_small",
                "WARNING:betaH_mismatch",
                "WARNING:betaH_nonfinite",
                "WARNING:nucleationRate_nonexponential",
            ],
        )
        derived = TransitionObservables._finalize_results(obs, ctx)

        self.assertTrue(derived["WARNING:betaH_small"])
        self.assertTrue(derived["WARNING:betaH_very_small"])
        self.assertTrue(derived["WARNING:betaH_mismatch"])
        self.assertFalse(derived["WARNING:betaH_nonfinite"])
        self.assertFalse(derived["WARNING:nucleationRate_nonexponential"])

    def test_finalize_results_detects_nonfinite_and_nonexponential_cases(self):
        obs = TransitionObservables.__new__(TransitionObservables)

        ctx = types.SimpleNamespace(
            derived_params={
                "betaH_S3": -1.0,
                "betaH_RH": np.nan,
            },
            derived_param_names=[
                "WARNING:betaH_small",
                "WARNING:betaH_very_small",
                "WARNING:betaH_mismatch",
                "WARNING:betaH_nonfinite",
                "WARNING:nucleationRate_nonexponential",
            ],
        )
        derived = TransitionObservables._finalize_results(obs, ctx)

        self.assertFalse(derived["WARNING:betaH_small"])
        self.assertFalse(derived["WARNING:betaH_very_small"])
        self.assertFalse(derived["WARNING:betaH_mismatch"])
        self.assertTrue(derived["WARNING:betaH_nonfinite"])
        self.assertTrue(derived["WARNING:nucleationRate_nonexponential"])

    def test_rh_without_splines_does_not_become_negative(self):
        obs = TransitionObservables.__new__(TransitionObservables)
        ctx = types.SimpleNamespace(
            derived_params={
                "WARNING:no_perc_splines": True,
                "v_wall": 1.0,
                "c_s": 1 / np.sqrt(3),
            },
            derived_param_names=["RH"],
            verbose=False,
            outdict={},
            pot=object(),
            phase_symmetric=object(),
            phase_broken=object(),
            PercolationConf=types.SimpleNamespace(time_temperature_mode="bag", f_perc=0.28957),
        )
        percolation = types.SimpleNamespace(
            Tperc=1.0,
            Sint=None,
            TSYM=None,
            Hint=None,
            Pint=None,
            entropyInt=None,
            coolingInt=None,
        )

        TransitionObservables._compute_beta_and_separation(obs, ctx, percolation, 0.5, 2.0)

        self.assertTrue(np.isnan(ctx.derived_params["RH"]))

if __name__ == "__main__":
    unittest.main()
