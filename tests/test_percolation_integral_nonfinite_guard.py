"""The percolation integral must not hand non-finite values to its interpolants.

``percIntegralODE`` builds ``log_gamma_source`` and ``log_transport`` and clamps them
with ``np.maximum(..., _LOG_GAMMA_SOURCE_FLOOR)``. That clamp removes ``-inf``, which is
how an underflowing nucleation rate shows up, but ``np.maximum`` propagates ``NaN`` and
leaves ``+inf`` untouched. Both then reach ``PchipInterpolator`` and surface as
``ValueError: `y` must contain only finite values``, which the pipeline reports as a
bare error -1 with no typed TL error and no diagnostic.

``+inf`` is reachable from the inputs the adaptive controller actually produces: it
extends the cold boundary toward T -> 0, so ``H * T * 3 c_s^2`` can underflow to exactly
zero and ``-log(0)`` is ``+inf``.

Part of TransitionListener v2
Documentation: https://tasillo.de/TransitionListener/
"""

from __future__ import annotations

import unittest

import numpy as np

from transitionlistener import bubbledynamics as bd


def _ordinary_grid(n: int = 32):
    """A descending temperature grid with values that cause no numerical trouble."""
    T = np.linspace(50.0, 20.0, n)
    H = np.full(n, 1.0e-16)
    S = np.linspace(200.0, 100.0, n)
    cs2 = np.full(n, 1.0 / 3.0)
    a = np.linspace(1.0, 1.4, n)
    return T, H, S, cs2, a


class PercolationIntegralNonFiniteGuardTests(unittest.TestCase):
    """The floor clamp has to cover every non-finite case, not only -inf."""

    def test_ordinary_grid_still_integrates(self):
        """Guard the guard: the control case must keep working."""
        T, H, S, cs2, a = _ordinary_grid()
        out = bd.percIntegralODE(T, H, S, 1.0, sound_speed_sq=cs2, scale_factor=a)
        values = np.asarray(out if not isinstance(out, tuple) else out[0], dtype=float)
        self.assertTrue(np.all(np.isfinite(values)))

    def test_cold_tail_that_underflows_does_not_raise_a_bare_valueerror(self):
        """T -> 0 with a tiny Hubble rate must not crash inside SciPy.

        This is the shape of grid the cold-boundary extension walks into. The product
        ``H * T * 3 c_s^2`` underflows to zero, so ``-log(...)`` is ``+inf`` and the
        clamp lets it through.
        """
        n_head = 24
        T = np.concatenate([np.linspace(50.0, 20.0, n_head),
                            np.geomspace(1.0e-2, 1.0e-40, 8)])
        H = np.concatenate([np.full(n_head, 1.0e-16),
                            np.geomspace(1.0e-20, 1.0e-290, 8)])
        S = np.concatenate([np.linspace(200.0, 100.0, n_head), np.full(8, 100.0)])
        cs2 = np.full(T.size, 1.0 / 3.0)
        a = np.linspace(1.0, 2.0, T.size)

        # the premise of the test: the product really does underflow here
        self.assertEqual(H[-1] * T[-1] * 3.0 * cs2[-1], 0.0)

        try:
            out = bd.percIntegralODE(T, H, S, 1.0, sound_speed_sq=cs2, scale_factor=a)
        except ValueError as exc:  # pragma: no cover - this is the bug being fixed
            self.fail(
                "percIntegralODE let a non-finite value reach its interpolant: "
                f"{exc}. The floor clamp must cover +inf and NaN, not only -inf."
            )
        values = np.asarray(out if not isinstance(out, tuple) else out[0], dtype=float)
        self.assertTrue(
            np.all(np.isfinite(values)),
            "the percolation integral returned non-finite values",
        )

    def test_nan_in_the_action_does_not_reach_the_interpolant(self):
        """A NaN action entry must be floored like -inf, not propagated.

        ``np.maximum(nan, floor)`` is ``nan``, so a single bad action sample would
        otherwise poison the whole interpolant.
        """
        T, H, S, cs2, a = _ordinary_grid()
        S = S.copy()
        S[5] = np.nan
        try:
            out = bd.percIntegralODE(T, H, S, 1.0, sound_speed_sq=cs2, scale_factor=a)
        except ValueError as exc:  # pragma: no cover - this is the bug being fixed
            self.fail(f"a NaN action sample reached the interpolant: {exc}")
        values = np.asarray(out if not isinstance(out, tuple) else out[0], dtype=float)
        self.assertTrue(np.all(np.isfinite(values)))


if __name__ == "__main__":
    unittest.main()
