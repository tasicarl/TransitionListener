"""A model that asks for a tighter path-deformation target must get it.

``generic_potential.__init__`` auto-tightens ``fRatioConv`` to 1e-2 for multi-field
models, because the 1d default of 2e-2 leaves the bounce action too jittery for
percolation. It guards that override with a sentinel so a value the *user* chose is
respected -- but the sentinel is read **after** ``setConfigParameters()``, so it already
holds whatever the model asked for. The equality test then always passes and a model's own
choice is overwritten.

Consequence, measured: the 2HDM high-accuracy model declares ``fRatioConv = 5e-3`` and
every run of it has been using 1e-2. At the error-17 outlier of the edge line
(lambda3 = 5.8164076664, T = 26.7116 GeV) the two settings give S_3/T of 211.518 and
212.086, a difference of 0.568, which is 0.247 orders of magnitude in the nucleation
rate. The whole purpose of the model variant is that precision, and it never took effect.

Part of TransitionListener v2
Documentation: https://tasillo.de/TransitionListener/
"""

from __future__ import annotations

import unittest
from pathlib import Path

from transitionlistener.helper_functions import load_potential

REPO = Path(__file__).resolve().parents[1]
MODEL = REPO / "models" / "TL_2HDM_BSMPT_highacc.py"

PARAMS = dict(lambda1=0.28894, lambda2=0.26237, lambda3=5.77, lambda4=-2.17494,
              lambda5=-2.2417, m12_sq_GeV2=1573.171, tan_beta=21.3949,
              yukawa_type=1, v_GeV=246.22)


def deform_params(pot):
    return pot.config.tracingConf.tunneling_params["deformation_deform_params"]


class ModelDeclaredFRatioConvTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not MODEL.exists():
            raise unittest.SkipTest(f"model file missing: {MODEL}")

    def test_a_model_that_declares_a_tighter_target_keeps_it(self):
        """The bug: the declared 5e-3 is replaced by the generic 1e-2."""
        cls_ = load_potential(str(MODEL), "R2HDM_HighAcc_NAct100_Conv1_FRatio005")
        pot = cls_(PARAMS, verbose=False)
        self.assertGreaterEqual(getattr(pot, "Ndim", 1), 2,
                                "this test only means anything for a multi-field model")
        self.assertAlmostEqual(
            float(deform_params(pot)["fRatioConv"]), 5.0e-3, places=12,
            msg="the model declared fRatioConv = 5e-3 in setConfigParameters and the "
                "multi-field auto-tighten overwrote it; the model's declared precision "
                "never reaches the bounce solver",
        )

    def test_the_auto_tighten_still_applies_to_a_model_that_declares_nothing(self):
        """Control: the override must keep working where it is wanted.

        The plain 2HDM does not set ``fRatioConv``, so it must still be tightened from
        the 1d default of 2e-2 to 1e-2 rather than left loose.
        """
        plain = REPO / "models" / "TL_2HDM_BSMPT.py"
        if not plain.exists():
            self.skipTest(f"model file missing: {plain}")
        cls_ = load_potential(str(plain), "R2HDM")
        pot = cls_(PARAMS, verbose=False)
        self.assertGreaterEqual(getattr(pot, "Ndim", 1), 2)
        self.assertAlmostEqual(
            float(deform_params(pot)["fRatioConv"]), 1.0e-2, places=12,
            msg="the multi-field auto-tighten stopped applying to a model that makes no "
                "choice of its own",
        )


if __name__ == "__main__":
    unittest.main()
