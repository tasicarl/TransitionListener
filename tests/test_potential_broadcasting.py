"""Vtot and its derivatives must honour the documented (X, T) broadcasting contract.

``generic_potential.Vtot`` documents that ``X.shape[:-1]`` and ``T.shape`` only
need to be broadcastable. Model mass functions allocate their output from the
field shape, and Vtot adds the temperature-dependent terms in place, so a
single field point evaluated at an array of temperatures used to crash
(GitHub issue: "R2HDM.boson_mass_function doesn't respect the documented
(X, T) broadcasting contract").
"""

from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path

import numpy as np

from transitionlistener.helper_functions import load_potential

REPO = Path(__file__).resolve().parents[1]

# One parameter point per model file shipped in models/ (from examples/*.yaml
# where available; the 2HDM point is the one from the issue report).
MODELS = {
    "2HDM": ("models/TL_2HDM.py", "R2HDM", dict(
        lambda1=0.006, lambda2=0.25, lambda3=8.27, lambda4=-2.55, lambda5=0.76,
        m12_sq_GeV2=14186.7, tan_beta=17.7, yukawa_type=1, v_GeV=246.21965079413735)),
    "conformal_U1": ("models/TL_conformal_dark_u1.py", "specific_potential",
                     dict(g=0.7, v_GeV=0.1, y=0.01)),
    "dark_U1": ("models/TL_dark_U1.py", "specific_potential",
                dict(g_tilde=2.65, l=1.5e-3, v_GeV=1e7)),
    "dark_U1_g": ("models/TL_dark_U1_g_parameterization.py", "specific_potential",
                  {"g": 0.5, "lambda": 1.31e-3, "v_GeV": 1e3}),
    "flipflop": ("models/TL_dark_flipflop.py", "DarkFlipFlop", dict(
        lambda0=0.005098, lambda1=0.002144, lambda12=0.003078,
        v_GeV=3.728330741601088, y=0.972319, gamma=0.7532)),
    "template": ("models/templatePotential.py", "TemplatePotenital", {}),
}


def build(name):
    path, cls, params = MODELS[name]
    with contextlib.redirect_stdout(io.StringIO()):  # the 2HDM prints its inputs
        return load_potential(str(REPO / path), cls)(params, verbose=False)


def loop(fn, X, T):
    """Evaluate fn(x, t) one (x, t) pair at a time on the broadcast grid."""
    X = np.asarray(X, float)
    T = np.asarray(T, float)
    shape = np.broadcast_shapes(X.shape[:-1], T.shape)
    Xb = np.broadcast_to(X, shape + X.shape[-1:])
    Tb = np.broadcast_to(T, shape)
    first = np.asarray(fn(Xb[(0,) * len(shape)], float(Tb[(0,) * len(shape)])))
    out = np.empty(shape + first.shape)
    for idx in np.ndindex(*shape):
        out[idx] = fn(Xb[idx], float(Tb[idx]))
    return out


class PotentialBroadcastingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pots = {name: build(name) for name in MODELS}

    def _cases(self, pot):
        x0 = np.ravel(np.asarray(pot.X0, float))[: pot.Ndim]
        T = np.max(np.abs(x0)) * np.array([0.10, 0.15, 0.20, 0.25])
        X3 = np.stack([0.2 * x0, 0.5 * x0, x0])
        return {
            "X(1,N) T(4)": (0.9 * x0[None, :], T),       # the reported crash
            "X(N) T(4)": (0.9 * x0, T),
            "X(3,1,N) T(4)": (X3[:, None, :], T),        # outer product
            "X(3,N) T(1)": (X3, T[:1]),
            "X(3,N) T()": (X3, T[0]),
        }

    def test_vtot_matches_pointwise_evaluation(self):
        for name, pot in self.pots.items():
            for daisy in ("ArnoldEspinosa", "Parwani", "off"):
                pot.daisy = daisy
                for label, (X, T) in self._cases(pot).items():
                    with self.subTest(model=name, daisy=daisy, shapes=label):
                        got = pot.Vtot(X, T)
                        ref = loop(pot.Vtot, X, T)
                        self.assertEqual(np.shape(got), ref.shape)
                        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0)
            pot.daisy = "ArnoldEspinosa"

    def test_mass_spectrum_and_derivatives_broadcast(self):
        for name, pot in self.pots.items():
            X, T = self._cases(pot)["X(N) T(4)"]
            checks = {
                "boson_massSq": lambda x, t: pot.boson_massSq(x, t)[0],
                "Vdaisy_from_X": pot.Vdaisy_from_X,
                "DVtot": pot.DVtot,
                "gradV": pot.gradV,
                "d2V": pot.d2V,
                "dgradV_dT": pot.dgradV_dT,
            }
            for label, fn in checks.items():
                with self.subTest(model=name, fn=label):
                    np.testing.assert_allclose(fn(X, T), loop(fn, X, T), rtol=1e-12, atol=0)


if __name__ == "__main__":
    unittest.main()
