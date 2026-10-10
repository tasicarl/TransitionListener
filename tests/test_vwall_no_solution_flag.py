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


def _run(returned_vw, wall_velocity="LTE", module=None, non_monotonic=None):
    """Drive _ensure_wall_velocity with a stubbed hydrodynamics solver.

    ``module`` selects the backend, because both carry this verdict and a flag added to
    one of the two copies is dead code on the other's executing path. ``non_monotonic``
    sets the attribute `max_al` leaves behind, or leaves it unset where it is None, which
    is what every path of `calcWallVelocityLTE` that returns before reaching `max_al`
    does.
    """
    if module is None:
        from transitionlistener import transitionObservables as module

    class _StubHydro:
        def __init__(self, *a, **k):
            if non_monotonic is not None:
                self.max_al_non_monotonic = non_monotonic

        def calcWallVelocityLTE(self, T):
            return returned_vw

    original = module.Hydrodynamics
    module.Hydrodynamics = _StubHydro
    try:
        derived = {}
        ctx = types.SimpleNamespace(
            derived_param_names={"v_wall"},
            derived_params=derived,
            GWconfig=types.SimpleNamespace(wall_velocity=wall_velocity),
            pot=None, phase_symmetric=None, phase_broken=None, verbose=False,
        )
        module.TransitionObservables._ensure_wall_velocity(None, ctx, 100.0)
        return derived
    finally:
        module.Hydrodynamics = original


def _backends():
    from transitionlistener import transitionObservables, transitionObservables_fixedstep
    return (transitionObservables, transitionObservables_fixedstep)


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


class BothBackendsCarryTheVerdictTests(unittest.TestCase):
    """The same stub, through both observables modules.

    `_ensure_wall_velocity` exists twice, and both copies call the same
    `Hydrodynamics.calcWallVelocityLTE`. A verdict written in one copy only leaves the
    other backend reporting the template default for a key that is registered for every
    run, which reads as "no non-solution" rather than "not looked at".
    """

    def test_zero_is_flagged_on_both_backends(self):
        for module in _backends():
            with self.subTest(module=module.__name__):
                self.assertIs(_run(0, module=module)["WARNING:v_wall_no_solution"], True)

    def test_an_ordinary_wall_is_not_flagged_on_either_backend(self):
        for module in _backends():
            with self.subTest(module=module.__name__):
                self.assertIs(_run(0.6, module=module)["WARNING:v_wall_no_solution"], False)


class BranchAmbiguityReachesTheObservablesTests(unittest.TestCase):
    """`max_al_non_monotonic` must become `WARNING:v_wall_branch_ambiguous`.

    `tests/test_max_al_searches_the_range.py` shows that `max_al` sets the attribute on a
    measured non-monotonic point. That is a different statement from the observable being
    written: the attribute is read through `getattr(..., False)` in both backends, and a
    read that is dropped leaves the flag permanently false with nothing to say so.
    """

    def test_a_non_monotonic_residual_is_reported_on_both_backends(self):
        for module in _backends():
            with self.subTest(module=module.__name__):
                derived = _run(0.7475, module=module, non_monotonic=True)
                self.assertIs(derived["WARNING:v_wall_branch_ambiguous"], True)

    def test_a_monotonic_residual_is_not_reported_on_either_backend(self):
        for module in _backends():
            with self.subTest(module=module.__name__):
                derived = _run(0.7475, module=module, non_monotonic=False)
                self.assertNotIn("WARNING:v_wall_branch_ambiguous", derived)

    def test_a_solver_that_never_reached_max_al_is_not_reported(self):
        """Every early return of `calcWallVelocityLTE` leaves the attribute unset."""
        for module in _backends():
            with self.subTest(module=module.__name__):
                derived = _run(1, module=module, non_monotonic=None)
                self.assertNotIn("WARNING:v_wall_branch_ambiguous", derived)

    def test_the_flag_is_registered_for_the_writer(self):
        from transitionlistener import config
        from transitionlistener.interface import samplers, output_schema
        self.assertIn("WARNING:v_wall_branch_ambiguous", config.all_observables)
        self.assertIn("WARNING:v_wall_branch_ambiguous", samplers.get_empty_result())
        self.assertIn("WARNING:v_wall_branch_ambiguous", output_schema.WARNING_COLUMN_ORDER)


if __name__ == "__main__":
    unittest.main(verbosity=2)
