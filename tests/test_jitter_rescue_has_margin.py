"""The rate-jitter rescue must actually tighten something.

``_try_action_jitter_tunneltight_rescue`` recomputes an outlier action with
``_tight_tunneling_params``. Those were built as ``min(current, constant)`` for
``converge_0`` and ``fRatioConv``, which leaves a model that already declares the
constants with **no tightening at all** - the rescue then recomputes the same action with
bit-identical settings and cannot change the residual.

That is the state of ``R2HDM_HighAcc_NAct100_Conv1_FRatio005``, whose name records that it
declares ``converge_0 = 1.0`` and ``fRatioConv = 5e-3``: exactly the two constants. Before
the ordering fix in ``generic_potential.__init__`` the model's 5e-3 was being overwritten
with 1e-2, so the rescue happened to have a factor-2 margin (worth 0.247 orders of
magnitude in the rate at the measured outlier, already too small against a 1.45 oom
residual). Fixing that ordering removed the margin entirely.

So the tightening has to be relative to whatever the run uses. Measured justification:
halving ``fRatioConv`` from 1e-2 to 5e-3 took the action jitter from 0.78 to 0.28 rms in
``S_3/T`` on the 2HDM edge line, so a further factor buys a comparable reduction, and the
cost is bounded because the rescue only recomputes the outlier temperatures it targets.

Part of TransitionListener v2
Documentation: https://tasillo.de/TransitionListener/
"""

from __future__ import annotations

import types
import unittest
from pathlib import Path

from transitionlistener import bubbledynamics as bd

REPO = Path(__file__).resolve().parents[1]
MODEL = REPO / "models" / "TL_2HDM_BSMPT_highacc.py"
PARAMS = dict(lambda1=0.28894, lambda2=0.26237, lambda3=5.77, lambda4=-2.17494,
              lambda5=-2.2417, m12_sq_GeV2=1573.171, tan_beta=21.3949,
              yukawa_type=1, v_GeV=246.22)


def fake_pot(**deform):
    """`_tight_tunneling_params` reads only the tunnelling-parameter dict."""
    return types.SimpleNamespace(
        config=types.SimpleNamespace(
            tracingConf=types.SimpleNamespace(
                tunneling_params={"deformation_deform_params": dict(deform),
                                  "other_setting": 7})))


def deform_of(pot):
    return bd._tight_tunneling_params(pot)["deformation_deform_params"]


class JitterRescueMarginTests(unittest.TestCase):

    def test_a_model_already_at_the_constants_still_gets_tightened(self):
        """The bug: identical settings in, identical settings out, rescue is a no-op."""
        pot = fake_pot(converge_0=1.0, fRatioConv=5.0e-3, maxiter=500)
        tight = deform_of(pot)
        self.assertLess(
            float(tight["fRatioConv"]), 5.0e-3,
            "the rescue would recompute the outlier action with the very settings that "
            "produced it, so it cannot reduce the jitter residual at all",
        )

    def test_the_real_high_accuracy_model_gets_a_margin(self):
        """End-to-end on the model this actually bites: no mocking."""
        if not MODEL.exists():
            self.skipTest(f"model file missing: {MODEL}")
        from transitionlistener.helper_functions import load_potential
        cls = load_potential(str(MODEL), "R2HDM_HighAcc_NAct100_Conv1_FRatio005")
        pot = cls(PARAMS, verbose=False)
        run = pot.config.tracingConf.tunneling_params["deformation_deform_params"]
        tight = deform_of(pot)
        self.assertLess(float(tight["fRatioConv"]), float(run["fRatioConv"]))

    def test_a_loose_model_is_still_brought_to_at_least_the_old_target(self):
        """Control: the historical behaviour is a floor, never loosened."""
        pot = fake_pot(converge_0=5.0, fRatioConv=2.0e-2)
        tight = deform_of(pot)
        self.assertLessEqual(float(tight["fRatioConv"]), 5.0e-3)
        self.assertLessEqual(float(tight["converge_0"]), 1.0)

    def test_an_already_strict_model_is_never_loosened(self):
        """Control: a run asking for more precision keeps it."""
        pot = fake_pot(converge_0=0.5, fRatioConv=1.0e-4)
        tight = deform_of(pot)
        self.assertLessEqual(float(tight["fRatioConv"]), 1.0e-4)
        self.assertLessEqual(float(tight["converge_0"]), 0.5)

    def test_unrelated_settings_are_left_alone(self):
        """Control: only the deformation targets are touched."""
        pot = fake_pot(converge_0=1.0, fRatioConv=5.0e-3, maxiter=500, startstep=0.002)
        tight = deform_of(pot)
        self.assertEqual(tight["maxiter"], 500)
        self.assertEqual(tight["startstep"], 0.002)
        self.assertEqual(
            bd._tight_tunneling_params(pot)["other_setting"], 7,
            "the rescue altered a tunnelling setting outside the deformation block",
        )


if __name__ == "__main__":
    unittest.main()
