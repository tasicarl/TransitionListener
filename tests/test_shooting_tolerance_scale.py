"""The profile integration must be held to a fraction of the bounce, not of the gap between minima.

`findProfile` integrates the bounce from its centre out to the metastable vacuum and holds that
integration to an absolute error `epsabs`. That error was `phitol` times `delta_phi`, the whole
gap between the two minima, while the field range the integration actually covers is the
bounce's own excursion, `|phi(0) - phi_metaMin|`.

The two are the same thing on an ordinary transition, where the bounce starts somewhere between
the minima. They are not on a supercooled one: the escape point sits a few `T` from the false
vacuum while the minima stay `v` apart, so the excursion shrinks with the temperature and the
gap does not. Measured on the conformal dark U(1) at g = 0.46, where the gap is 1000 internal
units at every temperature and the excursion is about 4.3 T, `phitol |delta_phi|` is 1.6e-7 of
the integrated feature at T_int = 0.1 and 2.4e-2 of it at 1e-6. The action then comes out low,
smoothly rather than noisily, and below T_int = 1e-7 the solve collapses to zero.
"""
import unittest

import numpy as np

from transitionlistener import tunneling1D


def _two_minima(gap, barrier_at, height):
    """A false vacuum at 0, a barrier near the origin, and a deep minimum at `gap`.

    `V` is a quartic whose derivative vanishes at 0, `barrier_at` and `gap`, so the shape near
    the origin -- which is all the bounce sees -- is held fixed while `gap` moves the far
    minimum. Depth is set by `height` so the bounce stays thick-walled.
    """
    def V(p):
        x = np.asanyarray(p, dtype=float)
        x = x[..., 0] if x.ndim and x.shape[-1] == 1 else x
        return height * (0.25 * x**4
                         - (barrier_at + gap) / 3.0 * x**3
                         + barrier_at * gap / 2.0 * x**2)

    def dV(p):
        x = np.asanyarray(p, dtype=float)
        x = x[..., 0] if x.ndim and x.shape[-1] == 1 else x
        return height * x * (x - barrier_at) * (x - gap)
    return V, dV


class EpsabsIsAFractionOfTheBounce(unittest.TestCase):

    @staticmethod
    def _epsabs_and_excursion(gap, phitol=1e-10):
        """Run findProfile and report the `epsabs` it used and the excursion it covered."""
        V, dV = _two_minima(gap, barrier_at=1.0, height=1.0e-3)
        # reflected so the deep minimum is at coordinate 0, which is the orientation the
        # solver wants and the one the converged reference uses
        inst = tunneling1D.SingleFieldInstanton(
            0.0, gap, lambda x: V(gap - np.asarray(x)), lambda x: -dV(gap - np.asarray(x)))
        seen = {}
        real = tunneling1D.SingleFieldInstanton.integrateProfile

        def spy(self, r0, y0, dr0, epsfrac, epsabs, drmin, rmax, *a, **k):
            seen["epsabs"] = float(np.asarray(epsabs, dtype=float)[0])
            seen["phi0"] = float(y0[0])
            return real(self, r0, y0, dr0, epsfrac, epsabs, drmin, rmax, *a, **k)

        tunneling1D.SingleFieldInstanton.integrateProfile = spy
        try:
            inst.findProfile(xtol=1e-10, phitol=phitol)
        finally:
            tunneling1D.SingleFieldInstanton.integrateProfile = real
        return seen["epsabs"], abs(seen["phi0"] - inst.phi_metaMin)

    def test_it_is_phitol_times_the_excursion(self):
        phitol = 1e-10
        eps, excursion = self._epsabs_and_excursion(gap=50.0, phitol=phitol)
        self.assertAlmostEqual(
            eps / excursion, phitol, delta=0.25 * phitol,
            msg=f"epsabs is {eps:.6g} against an excursion of {excursion:.6g}, a ratio of "
                f"{eps/excursion:.3g} rather than phitol = {phitol:g}")

    def test_it_does_not_follow_the_distant_minimum(self):
        """Moving the far minimum away must not loosen the integration of the bounce.

        The bounce never reaches that minimum, so nothing about it may change `epsabs`. Under
        the rule this replaces, `epsabs` was proportional to the gap and grew with it.
        """
        ratios = {}
        for gap in (50.0, 500.0):
            eps, excursion = self._epsabs_and_excursion(gap=gap)
            ratios[gap] = eps / excursion
        self.assertAlmostEqual(
            ratios[500.0], ratios[50.0], delta=0.25 * ratios[50.0],
            msg=f"a tenfold more distant minimum changed epsabs relative to the bounce: "
                f"{ratios}")


if __name__ == "__main__":
    unittest.main()
