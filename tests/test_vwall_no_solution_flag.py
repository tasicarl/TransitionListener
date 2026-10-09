"""Change 26: a wall velocity of exactly zero is a verdict, and must be reported as one.

`hydrodynamics.find_vw` returns exactly 0 when alpha_N falls below the threshold where a wall
solution exists at all. Every kinetic energy fraction is proportional to the wall velocity, so
the spectrum is then identically zero and LISA_SNR comes out exactly 0.0, which reads as a
real prediction of no signal. Measured at lambda3 = 5.5475 on the 2HDM line: alpha = 0.0347,
the largest of its neighbours, and betaH_RH = 318, the smallest, both of which should *raise*
the SNR; the points at +-0.00025 give 1.83 and 0.058.

The trap this test guards: v_wall == 1 is the *normal* outcome on a supercooled line -- 125 of
126 resolved points, via the documented runaway branches -- so an implementation that flagged
"unusual" wall velocities, or anything outside (0, 1), would fire on almost every point. Only
the zero is a non-solution.
"""
import types
import unittest


def _run(returned_vw, wall_velocity="LTE"):
    """Drive _ensure_wall_velocity with a stubbed hydrodynamics solver."""
    from transitionlistener import transitionObservables as tobs

    class _StubHydro:
        def __init__(self, *a, **k):
            pass

        def calcWallVelocityLTE(self, T):
            return returned_vw

    original = tobs.Hydrodynamics
    tobs.Hydrodynamics = _StubHydro
    try:
        derived = {}
        ctx = types.SimpleNamespace(
            derived_param_names={"v_wall"},
            derived_params=derived,
            GWconfig=types.SimpleNamespace(wall_velocity=wall_velocity),
            pot=None, phase_symmetric=None, phase_broken=None, verbose=False,
        )
        tobs.TransitionObservables._ensure_wall_velocity(None, ctx, 100.0)
        return derived
    finally:
        tobs.Hydrodynamics = original


class VWallNoSolutionTests(unittest.TestCase):
    KEY = "WARNING:v_wall_no_solution"

    def test_zero_is_flagged(self):
        d = _run(0)
        self.assertEqual(d["v_wall"], 0)
        self.assertIs(d[self.KEY], True)

    def test_runaway_one_is_not_flagged(self):
        """v_wall = 1 is the documented runaway branch and the common case here."""
        d = _run(1)
        self.assertIs(d[self.KEY], False)

    def test_ordinary_subluminal_wall_is_not_flagged(self):
        d = _run(0.6624337147913631)
        self.assertIs(d[self.KEY], False)

    def test_flag_is_always_present_so_downstream_can_rely_on_it(self):
        for vw in (0, 0.5, 1):
            self.assertIn(self.KEY, _run(vw), f"key missing for v_wall = {vw}")

    def test_negative_is_flagged_too(self):
        """Not reachable from find_vw today, but it is no more a wall velocity than zero."""
        self.assertIs(_run(-0.1)[self.KEY], True)

    def test_the_default_is_false_not_missing(self):
        """The observables template must carry the key, so an unflagged point says so."""
        import re
        from transitionlistener import transitionObservables as tobs
        src = open(tobs.__file__).read()
        self.assertRegex(src, re.escape('"WARNING:v_wall_no_solution": False'))

    def test_config_knows_the_label(self):
        from transitionlistener import config
        src = open(config.__file__).read()
        self.assertIn("WARNING:v_wall_no_solution", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
