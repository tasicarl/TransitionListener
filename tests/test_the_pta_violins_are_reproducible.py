"""The violin markers must be the same on every call.

The markers are shaped from 100000 samples drawn from the free spectrum of one frequency
bin. Two builds of one figure from the same inputs must give the same marker outlines, and
a build must leave the global NumPy random state as it found it, so that a caller's own
sequence of random numbers is unaffected.

Part of TransitionListener v2
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""
import unittest

import numpy as np


def _violin_outlines(**kwargs):
    """The vertices of the violins drawn on a throw-away figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from transitionlistener import plots

    fig, ax = plt.subplots()
    try:
        plots.add_violins(ax, nfreq=2, **kwargs)
        return [np.asarray(path.vertices, dtype=float)
                for collection in ax.collections
                for path in collection.get_paths()]
    finally:
        plt.close(fig)


class ThePTAViolinsAreReproducibleTests(unittest.TestCase):

    def test_two_calls_draw_the_same_violins(self):
        first, second = _violin_outlines(), _violin_outlines()
        self.assertEqual(len(first), len(second))
        self.assertGreater(len(first), 0)
        for i, (a, b) in enumerate(zip(first, second)):
            with self.subTest(violin=i):
                self.assertEqual(a.shape, b.shape)
                np.testing.assert_array_equal(a, b)

    def test_the_global_random_state_does_not_reach_the_violins(self):
        """A caller that drew random numbers before must get the same figure."""
        np.random.seed(1)
        first = _violin_outlines()
        np.random.seed(2)
        np.random.random(1000)
        second = _violin_outlines()
        for i, (a, b) in enumerate(zip(first, second)):
            with self.subTest(violin=i):
                np.testing.assert_array_equal(a, b)

    def test_the_call_leaves_the_global_random_state_untouched(self):
        """A caller's own sequence of random numbers must survive the call.

        Drawing the samples through the module-level generator after seeding it would
        give reproducible violins and still reset the state the caller draws from.
        """
        before = np.random.get_state()
        try:
            np.random.seed(7)
            state_before_call = np.random.get_state()
            _violin_outlines()
            state_after_call = np.random.get_state()

            self.assertEqual(state_before_call[0], state_after_call[0])
            np.testing.assert_array_equal(state_before_call[1], state_after_call[1])
            self.assertEqual(state_before_call[2:], state_after_call[2:])
        finally:
            np.random.set_state(before)

    def test_another_seed_draws_another_sample(self):
        """The default is one choice of seed, not a violin that ignores its samples."""
        default = _violin_outlines()
        other = _violin_outlines(seed=12345)
        self.assertEqual(len(default), len(other))
        differs = any(a.shape != b.shape or not np.array_equal(a, b)
                      for a, b in zip(default, other))
        self.assertTrue(differs,
                        "two seeds gave the same markers, so the samples are not used")


if __name__ == "__main__":
    unittest.main()
