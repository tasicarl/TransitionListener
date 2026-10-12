"""A phase's interpolated minima must stay inside the range the tracer established.

`PhaseInfo` interpolates the traced minima against temperature and the bounce solver reads the
result through `valAt`, so an interpolation that overshoots hands the solver a field value the
tracer never found. An interpolating cubic overshoots between unevenly spaced nodes, and the
tracer's nodes are very unevenly spaced: it clusters them where it steps small and leaves gaps
where it steps large.

The fixture below is the symmetric phase the tracer leaves on the 2HDM benchmark at
lambda3 = 5.8123128723, reduced to the 19 nodes between 26 and 69 GeV and rounded to the
0.1 MeV the measurement printed. On the full 63-node trace, whose spacings run from
2.5e-4 GeV inside the clusters to tens of GeV across the gaps, `splprep(..., s=0, k=3)` reaches
15.47 GeV at T = 38.5575; on these 19 rounded nodes it reaches 10.85 GeV at T = 38.56 and
68.59 GeV at T = 29.66. Either way no node is further than 0.118 GeV from the origin, and the
bounce tunnelled from the displaced point, over a shorter path and a lower barrier, returning an
action 4.7 low in S3/T.

These tests drive `PhaseInfo` itself, not a local copy of the interpolation.
"""
import unittest

import numpy as np

from transitionlistener.phases import PhaseInfo


# the measured nodes, in the tracer's own order, which is ascending in temperature
NODE_T = np.array([
    26.3973, 35.1965, 37.5770, 38.1831, 38.7892, 38.7894, 38.7897, 38.7901, 38.7905,
    38.7920, 38.8046, 38.8550, 39.0563, 39.4589, 40.2642, 41.8748, 45.0959, 51.5382, 68.7176,
])
NODE_X = np.array([
    [0.0012, 0.0013], [-0.0015, -0.0003], [0.0008, 0.0011], [-0.0204, -0.0117],
    [-0.0005, -0.0003], [-0.0005, -0.0003], [0.1011, 0.0606], [0.1000, 0.0599],
    [0.0991, 0.0593], [0.0932, 0.0558], [0.0671, 0.0401], [0.0409, 0.0248],
    [0.0237, 0.0145], [0.0156, 0.0082], [0.0102, 0.0046], [0.0068, 0.0024],
    [0.0045, 0.0010], [0.0030, 0.0003], [0.0019, -0.0001],
])
LARGEST_NODE = float(np.max(np.linalg.norm(NODE_X, axis=1)))   # 0.1178 GeV


def _phase(T=NODE_T, X=NODE_X):
    """A PhaseInfo on the given trace, with the derivative the tracer would have supplied."""
    dXdT = np.gradient(X, T, axis=0)
    return PhaseInfo(0, X, T, dXdT)   # the signature is (key, X, T, dXdT)


class ThePhaseStaysInsideItsOwnTrace(unittest.TestCase):
    def test_it_does_not_leave_the_range_of_the_traced_values(self):
        """The claim, on the measured nodes: no excursion beyond what the tracer found."""
        phase = _phase()
        grid = np.linspace(NODE_T.min(), NODE_T.max(), 4001)
        values = np.asarray([phase.valAt(float(t)) for t in grid])
        norms = np.linalg.norm(values, axis=1)
        self.assertLessEqual(
            float(norms.max()), LARGEST_NODE * 1.01,
            f"the phase reached {norms.max():.4g} GeV where its largest traced node is "
            f"{LARGEST_NODE:.4g} GeV")
        # the guarantee the interpolator actually gives is per component: each one stays
        # inside the range of its own traced values, which is what bounds the norm above.
        for j in range(NODE_X.shape[1]):
            self.assertGreaterEqual(float(values[:, j].min()), float(NODE_X[:, j].min()) - 1e-12,
                                    f"component {j} went below every traced value")
            self.assertLessEqual(float(values[:, j].max()), float(NODE_X[:, j].max()) + 1e-12,
                                 f"component {j} went above every traced value")

    def test_the_excursion_the_cubic_made_is_gone(self):
        """At T = 38.56 specifically, where the interpolating cubic returned 10.85 GeV on these
        nodes and 15.47 GeV on the full trace they were taken from."""
        value = np.linalg.norm(np.asarray(_phase().valAt(38.56)))
        self.assertLess(value, 1.0,
                        f"the symmetric phase sits at {value:.4g} GeV, not at the origin")

    def test_the_fixture_really_is_the_hard_case(self):
        """Without this the two tests above could pass on any well-behaved trace."""
        spacings = np.diff(np.sort(NODE_T))
        self.assertLess(spacings.min(), 1e-3, "the fixture has no near-duplicate nodes")
        self.assertGreater(spacings.max() / spacings.min(), 1e3,
                           "the fixture's spacings are not unevenly distributed")
        self.assertLess(LARGEST_NODE, 0.2, "the fixture's nodes are not near the origin")

    def test_it_still_passes_through_every_traced_node(self):
        """Shape preserving, not smoothing: the traced values are data, not suggestions."""
        phase = _phase()
        for t, x in zip(NODE_T, NODE_X):
            got = np.asarray(phase.valAt(float(t)), dtype=float).ravel()
            self.assertTrue(np.allclose(got, x, rtol=0, atol=1e-9),
                            f"at T = {t} the phase gives {got}, not the traced {x}")


class TheInterfaceValAtOffersIsUnchanged(unittest.TestCase):
    """The interpolator behind `valAt` changed; what `valAt` returns to the 114 call sites in
    `src` must not. Exactly one of them passes `deriv`, and none passes a two-dimensional `T`,
    so a regression in either would be found by its consequences and not by a failure."""

    def test_a_two_point_trace_still_interpolates(self):
        """The shortest trace the tracer can leave: the result is the line between them."""
        T = np.array([40.0, 50.0])
        X = np.array([[1.0, 2.0], [3.0, 6.0]])
        got = np.asarray(_phase(T, X).valAt(45.0), dtype=float).ravel()
        self.assertTrue(np.allclose(got, [2.0, 4.0], atol=1e-9), f"got {got}")

    def test_a_trace_of_one_temperature_is_refused(self):
        """Nothing to interpolate. The old spline refused it too, with `m > k must hold`.

        Built without the helper: `np.gradient` cannot take a derivative of one point either,
        and the point here is that `PhaseInfo` is what refuses the trace. The message is
        asserted because the interpolator would refuse it too, with one of its own, so only
        the message distinguishes the check in `_build_interpolator` from no check at all.
        """
        with self.assertRaisesRegex(ValueError, "at least two traced temperatures"):
            PhaseInfo(0, np.array([[1.0, 2.0]]), np.array([40.0]), np.zeros((1, 2)))

    def test_the_derivative_is_still_available(self):
        """`valAt(deriv=1)` is the documented way to get dX/dT and must keep working."""
        T = np.linspace(40.0, 60.0, 11)
        X = np.stack([2.0 * T, -3.0 * T], axis=1)
        d = np.asarray(_phase(T, X).valAt(50.0, deriv=1), dtype=float).ravel()
        self.assertTrue(np.allclose(d, [2.0, -3.0], rtol=1e-6),
                        f"dX/dT came back as {d}, not [2, -3] on a linear trace")

    def test_the_order_asked_for_is_the_order_returned(self):
        """The test above asks only for the first order, which an implementation that always
        took the first derivative would answer correctly however it read its argument.

        Here each order is compared against a central difference of the one below it, on a
        curved trace and at a temperature inside one piece of the piecewise cubic, so that the
        difference is taken on a single polynomial.
        """
        T = np.arange(0.0, 11.0)
        X = np.stack([T ** 3 / 100.0, -T ** 2 / 10.0], axis=1)
        phase = _phase(T, X)
        probe, h = 5.5, 1e-4
        got = [np.asarray(phase.valAt(probe, deriv=d), dtype=float).ravel() for d in range(4)]
        for d in (1, 2, 3):
            fd = (np.asarray(phase.valAt(probe + h, deriv=d - 1), dtype=float).ravel()
                  - np.asarray(phase.valAt(probe - h, deriv=d - 1), dtype=float).ravel()) / (2 * h)
            self.assertTrue(np.allclose(got[d], fd, rtol=1e-4, atol=1e-8),
                            f"deriv={d} gave {got[d]}, the difference of deriv={d - 1} gives {fd}")
            # and the orders differ here, so the comparison above can fail
            self.assertFalse(np.allclose(got[d], got[d - 1], rtol=1e-3, atol=1e-8),
                             f"deriv={d} and deriv={d - 1} agree on this trace; the fixture "
                             f"cannot tell the orders apart")
        # the fourth derivative of a piecewise cubic is zero
        self.assertTrue(np.allclose(np.asarray(phase.valAt(probe, deriv=4), dtype=float), 0.0))

    def test_the_result_is_one_field_vector_per_temperature(self):
        """An orientation regression here would be silent and would corrupt every caller.

        The number of fields differs from the number of temperatures asked for, so a transposed
        result cannot be mistaken for the right one. The two-dimensional case is deliberately
        not square, which a transpose of either the argument or the result would change.
        """
        T = np.linspace(40.0, 60.0, 9)
        X = np.stack([T, 2.0 * T, -T], axis=1)        # three fields, nine temperatures
        phase = _phase(T, X)
        self.assertEqual(np.asarray(phase.valAt(45.0)).shape, (3,))
        probes = np.array([42.0, 44.0, 47.0, 50.0, 53.0, 58.0])
        rows = np.asarray(phase.valAt(probes))
        self.assertEqual(rows.shape, (6, 3))
        for i, t in enumerate(probes):
            self.assertTrue(np.allclose(rows[i], np.asarray(phase.valAt(float(t)))),
                            f"row {i} is not the value at T = {t}")
        # a temperature array of any shape keeps its shape, with the fields appended
        block = probes.reshape(2, 3)
        got = np.asarray(phase.valAt(block))
        self.assertEqual(got.shape, (2, 3, 3))
        for i in range(2):
            for j in range(3):
                self.assertTrue(np.allclose(got[i, j], np.asarray(phase.valAt(float(block[i, j])))),
                                f"element ({i}, {j}) is not the value at T = {block[i, j]}")
        self.assertEqual(np.asarray(phase.valAt(probes, deriv=1)).shape, (6, 3))

    def test_it_extrapolates_outside_the_traced_range_rather_than_refusing(self):
        """`bubbledynamics` says so at its stencil: `valAt` must return a number past the ends.

        It clamps its stencils to the traced range rather than range-checking them, because it
        treats a raise from `valAt` as "this phase cannot be evaluated here"; refusing would be
        read as a missing phase, and a NaN would pass that test and poison the derivative.
        """
        phase = _phase()
        span = NODE_T.max() - NODE_T.min()
        for t in (NODE_T.min() - 1e-6 * span, NODE_T.min() - 1e-2 * span,
                  NODE_T.max() + 1e-6 * span, NODE_T.max() + 1e-2 * span):
            got = np.asarray(phase.valAt(float(t)), dtype=float)
            self.assertTrue(np.all(np.isfinite(got)),
                            f"valAt({t}) gave {got} outside the traced range")


class ReplacingTheTraceRebuildsIt(unittest.TestCase):
    def test_replace_trace_rebuilds_the_same_way(self):
        """`replaceTrace` has its own call to the build and must not be left on the old one."""
        phase = _phase(NODE_T, np.zeros_like(NODE_X))
        phase.replaceTrace(NODE_X, np.gradient(NODE_X, NODE_T, axis=0))
        # it rebuilt: the flat trace it was made with is gone and the new nodes are reproduced
        for t, x in zip(NODE_T, NODE_X):
            got = np.asarray(phase.valAt(float(t)), dtype=float).ravel()
            self.assertTrue(np.allclose(got, x, rtol=0, atol=1e-9),
                            f"after replaceTrace, T = {t} gives {got}, not the traced {x}")
        # and it rebuilt the same way: no overshoot between the new nodes either
        grid = np.linspace(NODE_T.min(), NODE_T.max(), 4001)
        norms = np.linalg.norm(np.asarray([phase.valAt(float(t)) for t in grid]), axis=1)
        self.assertLessEqual(
            float(norms.max()), LARGEST_NODE * 1.01,
            f"after replaceTrace the phase reached {norms.max():.4g} GeV where its largest "
            f"traced node is {LARGEST_NODE:.4g} GeV")


if __name__ == "__main__":
    unittest.main(verbosity=2)
