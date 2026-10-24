r"""The step-size error estimate of the bounce integrator, against the form it replaced.

`helper_functions.rkqs` decides whether a Runge-Kutta step is accepted by comparing the
step's error estimate against the absolute and fractional tolerances, and it takes the
smaller of the two comparisons for each component and the largest over the components. It
built that number by putting two arrays in a Python list for `np.min` to stack and reduce,
and then called `np.nan_to_num` on the single number that came out. That costs 23.9
microseconds against 4.7 for the form below, measured over 200000 repetitions, and the
bounce solver runs it once per step, 78658 times for six bounces of the real two Higgs
doublet model, which is why it is worth writing differently.

What the change may not do is return a different number. The tests here compare the two
forms on the cases where they could differ: a `nan` or an infinity in the error estimate,
a field value at zero, and a tolerance at zero or given per component.

Part of TransitionListener v2
"""
import unittest

import numpy as np

from transitionlistener import helper_functions as hf


def the_form_that_was_replaced(yerr, y, epsabs, epsfrac):
    """`rkqs`'s error estimate as it stood before the change, copied verbatim."""
    return np.nan_to_num(np.max(np.min([
        abs(yerr / epsabs), abs(yerr) / ((abs(y) + 1e-300) * epsfrac)
    ], axis=0)))


def the_form_in_use(yerr, y, epsabs, epsfrac):
    """The same arithmetic as `rkqs` performs now, so that the two can be compared.

    `rkqs` computes this inline and then uses it, so there is nothing to import. Keeping the
    copy next to the original is what makes the comparison below a comparison and not a
    restatement, and `test_the_copy_matches_the_source` reads the source to check that this
    copy has not drifted from it.
    """
    scaled = np.minimum(np.abs(yerr / epsabs),
                        np.abs(yerr) / ((np.abs(y) + 1e-300) * epsfrac))
    errmax = float(np.max(scaled))
    if errmax != errmax:
        return 0.0
    if errmax == np.inf:
        return hf._LARGEST_FLOAT
    if errmax == -np.inf:
        return -hf._LARGEST_FLOAT
    return errmax


#: Each case is (name, yerr, y, epsabs, epsfrac). The first is an ordinary accepted step on
#: the 2HDM benchmark; the rest are the inputs on which the two forms could disagree.
CASES = [
    ("an ordinary step", np.array([3.0e-9, -7.0e-10]), np.array([0.4, -1.2e-3]),
     1.0e-10, 1.0e-8),
    ("an error estimate of zero", np.zeros(2), np.array([0.4, -1.2e-3]), 1.0e-10, 1.0e-8),
    ("one component not a number", np.array([np.nan, -7.0e-10]),
     np.array([0.4, -1.0e-3]), 1.0e-10, 1.0e-8),
    ("every component not a number", np.array([np.nan, np.nan]),
     np.array([0.4, -1.0e-3]), 1.0e-10, 1.0e-8),
    ("one component infinite", np.array([np.inf, -7.0e-10]),
     np.array([0.4, -1.0e-3]), 1.0e-10, 1.0e-8),
    ("every component infinite", np.array([np.inf, np.inf]),
     np.array([0.4, -1.0e-3]), 1.0e-10, 1.0e-8),
    ("a negative infinity", np.array([-np.inf, -7.0e-10]),
     np.array([0.4, -1.0e-3]), 1.0e-10, 1.0e-8),
    ("the field at zero", np.array([3.0e-9, -7.0e-10]), np.zeros(2), 1.0e-10, 1.0e-8),
    ("no fractional tolerance", np.array([3.0e-9, -7.0e-10]),
     np.array([0.4, -1.0e-3]), 1.0e-10, 0.0),
    ("no absolute tolerance", np.array([3.0e-9, -7.0e-10]),
     np.array([0.4, -1.0e-3]), 0.0, 1.0e-8),
    ("tolerances given per component", np.array([3.0e-9, -7.0e-10]),
     np.array([0.4, -1.0e-3]), np.array([1.0e-10, 1.0e-12]),
     np.array([1.0e-8, 1.0e-6])),
    ("a single component", np.array([3.0e-9]), np.array([0.4]), 1.0e-10, 1.0e-8),
    ("eight components", np.linspace(1.0e-12, 1.0e-8, 8), np.linspace(-2.0, 2.0, 8),
     1.0e-10, 1.0e-8),
]


class TheTwoFormsReturnTheSameNumber(unittest.TestCase):

    def test_they_agree_on_every_case(self):
        for name, yerr, y, epsabs, epsfrac in CASES:
            with self.subTest(name):
                with np.errstate(all="ignore"):
                    was = float(the_form_that_was_replaced(yerr, y, epsabs, epsfrac))
                    now = float(the_form_in_use(yerr, y, epsabs, epsfrac))
                if was != was:
                    self.assertNotEqual(now, now, msg=f"{name}: one form gave a nan")
                else:
                    self.assertEqual(
                        was, now,
                        msg=f"{name}: the step would be judged on {now!r} where it used to "
                            f"be judged on {was!r}")

    def test_a_nan_is_sent_to_zero_and_so_accepts_the_step(self):
        """The behaviour `np.nan_to_num` gave and that the step acceptance rests on.

        `rkqs` accepts the step when the estimate is below 1, so sending a `nan` to zero
        accepts a step whose error could not be judged. That is what the solver has always
        done, and spelling the replacement out rather than calling `np.nan_to_num` is what
        makes it easy to change by accident.
        """
        with np.errstate(all="ignore"):
            value = the_form_in_use(np.array([np.nan, np.nan]), np.array([0.4, -1.0e-3]),
                                    1.0e-10, 1.0e-8)
        self.assertEqual(value, 0.0)
        self.assertLess(value, 1.0, msg="a step whose error is not a number is no longer "
                                        "accepted, which changes where the solver stops")

    def test_an_infinity_is_sent_to_the_largest_float_and_so_rejects_the_step(self):
        """Kept for faithfulness, and it cannot change what the solver does.

        `rkqs` uses the estimate twice, in `errmax < 1.0` and in `0.9*dt*errmax**-.25`. An
        infinity fails the first comparison and an infinity raised to minus a quarter is
        zero, while the largest float fails the same comparison and gives 1.45e-77; the step
        size that follows is `max(dttemp, dt*.1)`, or `min` for a negative step, and
        `dt*0.1` wins against both. So the two choices send the solver down the same path
        with the same numbers, and no behaviour of the solver can tell them apart. The
        branch is kept because it is what `np.nan_to_num` did and because a reader comparing
        the two forms would otherwise find a difference in the number they return, and it is
        checked by reading the source in `test_the_copy_matches_the_source`, since there is
        no behaviour to check it by.
        """
        with np.errstate(all="ignore"):
            value = the_form_in_use(np.array([np.inf, np.inf]), np.array([0.4, -1.0e-3]),
                                    1.0e-10, 1.0e-8)
        self.assertEqual(value, float(np.finfo(np.float64).max))
        self.assertGreater(value, 1.0)
        # The two choices give the same next step size, which is why the branch is
        # unobservable and the claim above is a claim about the source and not about
        # behaviour.
        for dt in (1.0e-3, -1.0e-3):
            with self.subTest(f"dt = {dt:g}"):
                with np.errstate(all="ignore"):
                    from_inf = 0.9 * dt * float(np.inf) ** -0.25
                    from_max = 0.9 * dt * hf._LARGEST_FLOAT ** -0.25
                pick = max if dt > 0 else min
                self.assertEqual(pick(from_inf, dt * 0.1), pick(from_max, dt * 0.1))

    def test_the_copy_matches_the_source(self):
        """The copy above has to be the arithmetic `rkqs` performs, or nothing is tested.

        Reading the source rather than calling `rkqs`, because the estimate is computed
        inside its loop and cannot be reached from outside without a derivative function and
        a whole integration behind it.
        """
        import inspect

        source = inspect.getsource(hf.rkqs)
        for fragment in ("np.minimum(np.abs(yerr / epsabs)",
                         "np.abs(yerr) / ((np.abs(y) + 1e-300) * epsfrac)",
                         "errmax = float(np.max(scaled))",
                         "if errmax != errmax:",
                         "errmax = 0.0",
                         "elif errmax == np.inf:",
                         "errmax = _LARGEST_FLOAT",
                         "elif errmax == -np.inf:",
                         "errmax = -_LARGEST_FLOAT"):
            self.assertIn(fragment, source,
                          msg=f"rkqs no longer contains {fragment!r}, so the copy in this "
                              f"test file is not the arithmetic under test")

    def test_the_largest_float_is_what_nan_to_num_uses(self):
        """The constant has to be the one `np.nan_to_num` would have put there."""
        with np.errstate(all="ignore"):
            self.assertEqual(hf._LARGEST_FLOAT, float(np.nan_to_num(np.inf)))


if __name__ == "__main__":
    unittest.main()
