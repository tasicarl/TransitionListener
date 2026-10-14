"""The samples that build the potential spline have to fall inside the barrier.

`SplinePath` interpolates `V` along the path from `V_spline_samples` evenly spaced points, plus
an overhang of 0.2 of the path length past each end. Both of those are fractions of the distance
between the two minima, while the barrier is a feature whose width is set by the temperature. In
a strongly supercooled model the barrier is a vanishing fraction of the path, so no sample lands
inside it and the spline interpolates straight over it.

Measured on `TL_conformal_dark_u1` at g = 0.54, where the minima are 1000 internal field units
apart and the spacing is therefore 0.1: the barrier top is at 0.118 at T_int = 0.1 and at
9.3e-5 at T_int = 1e-4, and the action came back infinite from T_int = 1e-2 down.

The samples are now counted, and re-placed towards the ends only when too few of them fall
inside the barrier. The condition matters as much as the remedy: the clustered grid reaches
spacing ratios of 1e-9, which an interpolating cubic spline does not take well on a potential
with the 2HDM's curvature. So a path whose barrier the uniform grid already carries must keep
exactly the samples it had.
"""
import unittest

import numpy as np

from transitionlistener import pathDeformation as pd


class TheUniformGridIsUnchanged(unittest.TestCase):
    """Whatever else happens, the grid for an ordinary path is the one it always was."""

    def test_it_is_the_formula_it_always_was(self):
        L, n = 240.0, 1000
        x = np.linspace(0, L, n)
        x_ext = np.arange(x[1], L * .2, x[1])
        expected = np.concatenate([-x_ext[::-1], x, L + x_ext])
        got = pd._uniform_V_samples(L, n)
        self.assertEqual(got.shape, expected.shape)
        self.assertTrue(np.array_equal(got, expected),
                        "the uniform grid changed, so every resolved path changed with it")

    def test_it_still_reaches_a_fifth_of_the_path_past_each_end(self):
        L = 240.0
        x = pd._uniform_V_samples(L, 1000)
        self.assertAlmostEqual(x.min(), -0.2 * L, delta=0.3)
        self.assertAlmostEqual(x.max(), 1.2 * L, delta=0.3)


class CountingTheSamplesInsideTheBarrier(unittest.TestCase):
    """`_barrier_is_resolved` is the condition, so it is tested on its own."""

    @staticmethod
    def _sampled(L, n, barrier_top, height, floor=0.0):
        """A path whose barrier is a bump of the given width and height near the far end."""
        x = pd._uniform_V_samples(L, n)
        y = np.where((x > L - 2.0 * barrier_top) & (x < L),
                     floor + height, floor)
        return x, y

    def test_a_barrier_many_samples_wide_is_resolved(self):
        L, n = 240.0, 1000
        x, y = self._sampled(L, n, barrier_top=20.0, height=1.0e7)
        self.assertTrue(pd._barrier_is_resolved(x, y, L))

    def test_a_barrier_narrower_than_the_spacing_is_not(self):
        L, n = 1000.0, 10000
        spacing = L / (n - 1)
        x, y = self._sampled(L, n, barrier_top=1e-4 * spacing, height=1.0e-18)
        self.assertFalse(pd._barrier_is_resolved(x, y, L),
                         "a barrier far below the sample spacing was called resolved")

    def test_the_threshold_is_the_number_of_samples_in_the_barrier(self):
        """Exactly the stated number is enough, one fewer is not."""
        L, n = 100.0, 1001
        spacing = L / (n - 1)
        x = pd._uniform_V_samples(L, n)
        for count, expected in ((pd._V_SAMPLE_MIN_IN_BARRIER, True),
                                (pd._V_SAMPLE_MIN_IN_BARRIER - 1, False)):
            width = (count + 0.5) * spacing
            y = np.where((x > L / 2) & (x < L / 2 + width), 1.0, 0.0)
            inside = int(np.count_nonzero(y[(x >= 0) & (x <= L)][1:-1] > 0.0))
            self.assertEqual(inside, count, f"the construction gave {inside}, not {count}")
            self.assertIs(pd._barrier_is_resolved(x, y, L), expected,
                          f"{count} samples inside the barrier was judged {not expected}")

    def test_a_potential_that_cannot_be_judged_is_left_alone(self):
        """Non-finite samples are not grounds for re-placing them."""
        L, n = 240.0, 1000
        x = pd._uniform_V_samples(L, n)
        y = np.full(x.shape, np.nan)
        self.assertTrue(pd._barrier_is_resolved(x, y, L),
                        "a grid that could not be judged was replaced anyway")


class TheClusteredGrid(unittest.TestCase):
    def test_it_is_usable_as_spline_knots(self):
        x = pd._clustered_V_samples(1000.0, 10000)
        self.assertTrue(np.all(np.diff(x) > 0.0),
                        "the knots are not strictly increasing, so splrep would refuse them")

    def test_it_resolves_a_barrier_the_uniform_grid_misses(self):
        L, n = 1000.0, 10000
        x = pd._clustered_V_samples(L, n)
        narrow = 9.3e-5                      # the barrier top at T_int = 1e-4, measured
        inside = int(np.count_nonzero((x > L - narrow) & (x < L)))
        self.assertGreaterEqual(inside, pd._V_SAMPLE_MIN_IN_BARRIER,
                                f"only {inside} samples fall within {narrow} of the far end")

    def test_its_overhang_follows_the_local_spacing_and_not_the_path_length(self):
        """The overhang is what dominates the error once the grid is clustered."""
        L = 1000.0
        x = pd._clustered_V_samples(L, 10000)
        reach = x.max() - L
        self.assertLess(reach, 1e-3 * L,
                        f"the overhang reaches {reach} past the end, which drags the spline "
                        f"towards values far above the barrier")
        self.assertGreater(reach, 0.0, "there is no margin past the end at all")

    def test_it_still_covers_the_whole_path(self):
        """The uniform half is kept, so a feature away from the ends is not lost."""
        L = 1000.0
        x = pd._clustered_V_samples(L, 10000)
        interior = x[(x >= 0.0) & (x <= L)]
        gaps = np.diff(interior)
        self.assertLess(gaps.max(), 2.5 * L / (10000 // 2),
                        f"the largest gap along the path is {gaps.max()}")


class TheReplacementIsJudgedBeforeItIsUsed(unittest.TestCase):
    """A replacement grid that does not resolve the barrier either must not be adopted.

    `_barrier_is_resolved` answers True without looking at anything on a grid with fewer than
    `_V_SAMPLE_MIN_FOR_CLUSTERING` in-path knots, because below that there are not enough knots
    for the re-placed spacing to be worth building. The clustered grid used to come out one
    knot under that threshold at the configured 1000 samples -- the two end grids meet at L/2
    and `np.unique` drops one of the pair -- so the check that is supposed to keep an
    inadequate replacement out of the spline accepted every one of them.
    """

    L = 1000.0

    def test_the_clustered_grid_carries_the_knots_it_was_asked_for(self):
        for n_samples in (pd._V_SAMPLE_MIN_FOR_CLUSTERING, 1001, 2000, 10000):
            with self.subTest(n_samples=n_samples):
                x = pd._clustered_V_samples(self.L, n_samples)
                n_in = int(np.count_nonzero((x >= 0.0) & (x <= self.L)))
                self.assertGreaterEqual(
                    n_in, n_samples,
                    f"the clustered grid has {n_in} in-path knots against the {n_samples} "
                    f"asked for, so `_barrier_is_resolved` judges it without looking")

    def test_a_clustered_grid_that_misses_the_barrier_is_not_resolved(self):
        """The safeguard itself: the count must be able to say no at the default setting."""
        x = pd._clustered_V_samples(self.L, pd._V_SAMPLE_MIN_FOR_CLUSTERING)
        y = np.where((x > 0.4999 * self.L) & (x < 0.5001 * self.L), 1.0, -1.0)
        self.assertLess(int(np.count_nonzero((x > 0.4999 * self.L) & (x < 0.5001 * self.L))),
                        pd._V_SAMPLE_MIN_IN_BARRIER,
                        "this grid does carry the barrier, so it is the wrong fixture")
        self.assertFalse(pd._barrier_is_resolved(x, y, self.L),
                         "a clustered grid with no samples in the barrier was accepted")


class SplinePathChoosesTheGrid(unittest.TestCase):
    """`SplinePath` must actually reach the clustered branch, and keep the barrier when it does.

    The two-field `fullTunneling` case in the other file runs 200000 samples over a path of
    length 1000, so its barrier of width 0.12 already holds thousands of uniform samples and
    never exercises this. The fixture here is sized the other way round: the barrier is narrow
    enough that the uniform grid carries fewer than two samples inside it.
    """

    @staticmethod
    def _potential(length, barrier_at, barrier_width, height):
        """Flat, with one narrow bump near one end and a deep minimum at the far end."""
        def V(p):
            q = np.asanyarray(p, dtype=float)
            x = q[..., 0] if q.ndim and q.shape[-1] >= 1 else q
            bump = height * np.exp(-0.5 * ((x - barrier_at) / barrier_width) ** 2)
            return -1.0e3 * (x / length) + bump
        return V

    def test_the_uniform_grid_really_is_blind_to_this_barrier(self):
        """The fixture check: without it the test below would pass for the wrong reason."""
        L, n = 1000.0, pd._V_SAMPLE_MIN_FOR_CLUSTERING
        V = self._potential(L, 0.5, 0.02, 5.0e2)
        x = pd._uniform_V_samples(L, n)
        y = V(x[:, np.newaxis])
        self.assertFalse(pd._barrier_is_resolved(x, y, L),
                         "the uniform grid already resolves this barrier")

    def test_the_clustered_grid_is_selected_and_keeps_the_barrier(self):
        L, n = 1000.0, pd._V_SAMPLE_MIN_FOR_CLUSTERING
        V = self._potential(L, 0.5, 0.02, 5.0e2)
        x = pd._clustered_V_samples(L, n)
        y = V(x[:, np.newaxis])
        self.assertTrue(pd._barrier_is_resolved(x, y, L),
                        "the clustered grid does not resolve the barrier either, so this "
                        "fixture cannot show that the branch is taken")
        path = pd.SplinePath(np.array([[0.0], [L]]), V, V_spline_samples=n, extend_to_minima=False)
        top = max(float(path.V(t)) for t in np.linspace(0.3, 0.7, 401))
        self.assertGreater(top, 2.0e2,
                           f"the spline the path uses flattened the barrier to {top}, so the "
                           f"clustered grid was not the one it kept")


if __name__ == "__main__":
    unittest.main()
