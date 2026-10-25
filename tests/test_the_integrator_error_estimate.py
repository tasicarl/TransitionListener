r"""The step-size error estimate of the bounce integrator, against the form it replaced.

`helper_functions.rkqs` decides whether a Runge-Kutta step is accepted by comparing the
step's error estimate against the absolute and fractional tolerances, and it takes the
smaller of the two comparisons for each component and the largest over the components. It
built that number by putting two arrays in a Python list for `np.min` to stack and reduce,
and then called `np.nan_to_num` on the single number that came out, which costs several
times what the same arithmetic written directly costs, once per step.

What the change may not do is return a different number, or change what `rkqs` does with it.
The arithmetic is `helper_functions._step_error_estimate`, which the tests below call, and
the step decisions are tested through `rkqs` itself with the Cash-Karp step replaced by one
that returns a chosen error estimate.

Part of TransitionListener v2
"""
import contextlib
import unittest

import numpy as np

from transitionlistener import helper_functions as hf


def the_form_that_was_replaced(yerr, y, epsabs, epsfrac):
    """`rkqs`'s error estimate as it stood before the change, copied verbatim."""
    return np.nan_to_num(np.max(np.min([
        abs(yerr / epsabs), abs(yerr) / ((abs(y) + 1e-300) * epsfrac)
    ], axis=0)))


@contextlib.contextmanager
def the_step_returns(yerr):
    """Give `rkqs` a Cash-Karp step whose error estimate is `yerr`.

    The estimate is formed inside `rkqs`'s own loop, so this is what makes the step
    decisions that follow from it reachable from a test.
    """
    original = hf._rkck
    dy = np.zeros_like(np.atleast_1d(np.asarray(yerr, dtype=float)))
    hf._rkck = lambda y, dydt, t, f, dt, args: (dy, yerr)
    try:
        yield
    finally:
        hf._rkck = original


def one_step(yerr, y, epsabs, epsfrac, t=1.0, dt=1.0e-3):
    """One `rkqs` step on that estimate, returning what `rkqs` returns."""
    with the_step_returns(yerr), np.errstate(all="ignore"):
        return hf.rkqs(y, np.zeros_like(y), t, lambda *a: np.zeros_like(y),
                       dt, epsfrac, epsabs)


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
    ("single precision", np.array([3.0e-9, -7.0e-10], dtype=np.float32),
     np.array([0.4, -1.2e-3], dtype=np.float32), np.float32(1.0e-10), np.float32(1.0e-8)),
    ("single precision, infinite", np.array([np.inf, np.inf], dtype=np.float32),
     np.array([0.4, -1.0e-3], dtype=np.float32), np.float32(1.0e-10), np.float32(1.0e-8)),
]


class TheTwoFormsReturnTheSameNumber(unittest.TestCase):

    def test_they_agree_on_every_case(self):
        for name, yerr, y, epsabs, epsfrac in CASES:
            with self.subTest(name):
                with np.errstate(all="ignore"):
                    was = float(the_form_that_was_replaced(yerr, y, epsabs, epsfrac))
                    now = float(hf._step_error_estimate(yerr, y, epsabs, epsfrac))
                if was != was:
                    self.assertNotEqual(now, now, msg=f"{name}: one form gave a nan")
                else:
                    self.assertEqual(
                        was, now,
                        msg=f"{name}: the step would be judged on {now!r} where it used to "
                            f"be judged on {was!r}")

    def test_the_infinity_substitution_keeps_the_dtype(self):
        """`np.nan_to_num` substitutes the input dtype's largest value, not float64's.

        A single precision error estimate is sent to 3.4028235e+38, and fixing the
        substitution at `float64` would return 1.7976931e+308 instead. The two differ by 270
        orders of magnitude, and both reject the step, so only the number says which was
        used.
        """
        single = np.array([np.inf, np.inf], dtype=np.float32)
        y = np.array([0.4, -1.0e-3], dtype=np.float32)
        with np.errstate(all="ignore"):
            value = hf._step_error_estimate(single, y, np.float32(1.0e-10),
                                            np.float32(1.0e-8))
            expected = float(np.nan_to_num(np.float32(np.inf)))
        self.assertEqual(value, expected)
        self.assertEqual(value, float(np.finfo(np.float32).max))
        self.assertNotEqual(value, float(np.finfo(np.float64).max))

    def test_a_nan_accepts_the_step_through_rkqs(self):
        """The behaviour `np.nan_to_num` gave, read off `rkqs` and not off the arithmetic.

        Sending a `nan` to zero accepts a step whose error could not be judged. Left as a
        `nan` it would fail `errmax < 1.0`, and `rkqs` would then halve its way down to
        `IntegrationError` instead of returning, which is what this distinguishes.
        """
        y = np.array([0.4, -1.0e-3])
        step = one_step(np.array([np.nan, np.nan]), y, 1.0e-10, 1.0e-8, dt=1.0e-3)
        self.assertEqual(step.Delta_t, 1.0e-3,
                         msg="the step was not accepted at the size it was tried at")
        self.assertAlmostEqual(step.dtnxt, 5.0e-3, places=12,
                               msg="an estimate of zero should give the fivefold step")

    def test_an_infinity_rejects_the_step_through_rkqs(self):
        """An infinite estimate fails `errmax < 1.0` at every step size.

        `rkqs` shrinks the step by ten each time and raises once it rounds down to nothing,
        which is the only outcome an estimate above 1 can have when it never improves.
        """
        y = np.array([0.4, -1.0e-3])
        with self.assertRaises(hf.IntegrationError):
            one_step(np.array([np.inf, np.inf]), y, 1.0e-10, 1.0e-8, t=1.0, dt=1.0e-3)

    def test_rkqs_judges_the_step_on_the_number_the_estimate_returns(self):
        """`rkqs` has to use the returned value, not a constant of its own.

        The tolerances below make the estimate exactly 0.5: the absolute comparison gives
        0.5 and the fractional one 5e+29, and the smaller is kept. `rkqs` accepts the step
        and, since 0.5 is above its 1.89e-4 threshold, offers `0.9 * dt * errmax**-0.2` as
        the next size. Any value other than 0.5 reaching that line gives a different number.
        """
        y = np.array([1.0])
        yerr = np.array([0.5])
        with np.errstate(all="ignore"):
            self.assertEqual(hf._step_error_estimate(yerr, y, 1.0, 1.0e-30), 0.5)
        dt = 1.0e-3
        step = one_step(yerr, y, 1.0, 1.0e-30, dt=dt)
        self.assertAlmostEqual(step.dtnxt, 0.9 * dt * 0.5 ** -0.2, places=12,
                               msg="the next step size does not follow from an estimate "
                                   "of 0.5, so `rkqs` judged the step on something else")


if __name__ == "__main__":
    unittest.main()
