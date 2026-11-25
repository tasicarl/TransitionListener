"""The mean bubble separation of a supercooled transition does not depend on how the bag limit is passed.

With ``Tmax / T`` of order 1e5 the nucleation history occupies the lowest few e-folds of the range. Passing the
bag relation as ``None`` or as ``ln s = 3 ln T`` is the same expansion history and must give the same separation.
"""
import unittest

import numpy as np

from transitionlistener import bubbledynamics as bd


class SupercooledSeparation(unittest.TestCase):
    def test_bag_limit_independent_of_how_it_is_passed(self):
        T, Tmax = 3e-6, 0.4
        # an action whose rate peaks between T and 2 T: S/T falls through ~130 there
        Sint = lambda x: np.asarray(x, float) * (100.0 + 30.0 * np.log(np.asarray(x, float) / T))
        Pint = lambda x: np.zeros_like(np.asarray(x, float))
        Hint = lambda x: np.full_like(np.asarray(x, float), 1e-24)
        r_none = bd.calcMeanBubbleSeparation(T, Tmax, Sint, Pint, Hint)
        r_log = bd.calcMeanBubbleSeparation(T, Tmax, Sint, Pint, Hint,
                                            logEntropyInt=lambda x: 3.0 * np.log(np.asarray(x, float)))
        self.assertTrue(np.isfinite(r_none) and np.isfinite(r_log))
        self.assertAlmostEqual(r_none / r_log, 1.0, delta=1e-4)

    def test_fixed_step_solver_agrees(self):
        from transitionlistener import bubbledynamics_fixedstep as fs
        T, Tmax = 3e-6, 0.4
        Sint = lambda x: np.asarray(x, float) * (100.0 + 30.0 * np.log(np.asarray(x, float) / T))
        Pint = lambda x: np.zeros_like(np.asarray(x, float))
        Hint = lambda x: np.full_like(np.asarray(x, float), 1e-24)
        r_ref = bd.calcMeanBubbleSeparation(T, Tmax, Sint, Pint, Hint,
                                            logEntropyInt=lambda x: 3.0 * np.log(np.asarray(x, float)))
        self.assertAlmostEqual(fs.calcMeanBubbleSeparation(T, Tmax, Sint, Pint, Hint) / r_ref, 1.0, delta=1e-4)


if __name__ == "__main__":
    unittest.main()
