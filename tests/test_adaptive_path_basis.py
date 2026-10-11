"""The path deformation enlarges its basis when the force ratio stops falling.

The deformed path is a spline over ``nb`` basis functions. If that basis cannot express the
correction the transverse equations of motion ask for, the force ratio stops falling and no
number of deformation steps helps.

Measured at lambda3 = 5.8205024605, T = 15.202392 GeV, against `fRatioConv = 0.005`:

    nb    fRatio     S_3/T
    10    0.00855   286.43   <- the sample the rate-jitter gate rejects on
    20    0.00799   299.92
    40    0.00581   299.62   <- on the smooth curve its neighbours occupy
    80    0.09998   300.78   <- the deformation is destabilised, 17x worse

The inner deformation reported convergence on all 20 outer iterations and never approached
its own 500-step limit, yet the force ratio sat on that floor, and the action at a fixed
temperature varied by 3.9% between runs as a result. Raising the outer iteration count from
20 to 60 moved the force ratio by nothing.

So the basis grows instead of the step count. The loop stops enlarging it once doing so
stops helping, and returns the last solve rather than the best one: substituting the
best-scoring output was tried and corrupted the profile, giving `P_max = 0` at three
couplings, so it was removed. With that, the measured point reaches `fRatio = 0.00354` --
under target for the first time -- in 7 outer iterations rather than exhausting 20, so it
converges and costs less.

Part of TransitionListener v2
Documentation: https://tasillo.de/TransitionListener/
"""

from __future__ import annotations

import numpy as np

from transitionlistener import pathDeformation as pd


def _two_field_potential():
    """A quartic with a metastable and a true minimum, enough to run the real loop."""
    def V(x, T=0.0):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        a, b = x[..., 0], x[..., 1]
        return (a ** 2 - 1.0) ** 2 + 0.6 * b ** 2 + 0.25 * a + 0.3 * a * b ** 2

    def dV(x, T=0.0):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        a, b = x[..., 0], x[..., 1]
        return np.stack([4.0 * a * (a ** 2 - 1.0) + 0.25 + 0.3 * b ** 2,
                         1.2 * b + 0.6 * a * b], axis=-1)

    return V, dV


def _run(force_ratios, deform_params=None, init_params=None):
    """Run the outer loop with a scripted force-ratio sequence, and report what it did.

    Two things are controlled on purpose, because what is under test is the plateau and
    basis logic rather than the deformation physics.

    `_compute_force_ratio` is scripted, so the decision sequence is the one the test means
    rather than whatever this potential happens to produce.

    `_should_stop` is forced False, so the loop actually iterates. The path through this
    symmetric potential is the straight line already, so the real deformation converges on
    the first iteration and the plateau branch is never reached -- which is how the first
    version of this test passed vacuously with an empty record.
    """
    V, dV = _two_field_potential()
    seen_nb, served = [], {"i": 0}
    original_ratio, original_stop = pd._compute_force_ratio, pd._should_stop

    def fake_ratio(pts, dphi, dVf, settings):
        seen_nb.append(int(settings.deformation_init_params.get("nb", 10)))
        i = served["i"]
        served["i"] += 1
        return force_ratios[min(i, len(force_ratios) - 1)]

    pd._compute_force_ratio = fake_ratio
    pd._should_stop = lambda iter_output: False
    try:
        res = pd.fullTunneling(
            [np.array([-1.0, 0.0]), np.array([1.0, 0.0])], V, dV,
            maxiter=6,
            deformation_deform_params=dict(deform_params or {"fRatioConv": 0.005,
                                                             "converge_0": 1.0}),
            # passed through, NOT copied: copying here would make the leak test vacuous,
            # since it is fullTunneling's own copy that the test is checking for
            deformation_init_params=init_params if init_params is not None else {},
        )
    except BaseException as exc:
        raise AssertionError(f"the scripted run raised {type(exc).__name__}: {exc}") from exc
    finally:
        pd._compute_force_ratio = original_ratio
        pd._should_stop = original_stop
    return res, seen_nb


def test_a_plateau_above_target_enlarges_the_basis():
    """The fix: no improvement while above target means the basis, not more steps."""
    # a flat sequence well above fRatioConv = 0.005
    _res, seen_nb = _run([0.02, 0.0199, 0.0198, 0.0197, 0.0196, 0.0195])
    assert max(seen_nb) > min(seen_nb), (
        f"the basis never grew on a plateau above target; nb stayed at {sorted(set(seen_nb))}")
    assert max(seen_nb) == min(seen_nb) * 2 ** pd._MAX_BASIS_DOUBLINGS, (
        f"the basis did not reach the cap of {pd._MAX_BASIS_DOUBLINGS} doublings: "
        f"{sorted(set(seen_nb))}")


def test_steady_improvement_does_not_enlarge_the_basis():
    """Control: a deformation that is still converging must be left alone."""
    _res, seen_nb = _run([0.02, 0.010, 0.005001, 0.004, 0.003, 0.002])
    assert len(set(seen_nb)) == 1, (
        f"the basis grew while the force ratio was still halving each iteration: "
        f"{sorted(set(seen_nb))}")


def test_a_ratio_already_under_target_does_not_enlarge_the_basis():
    """Control: nothing to fix when the path is already force-balanced."""
    _res, seen_nb = _run([0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
    assert len(set(seen_nb)) == 1, (
        f"the basis grew although the force ratio was under target throughout: "
        f"{sorted(set(seen_nb))}")


def test_the_basis_growth_is_capped():
    """Cost stays bounded: nb = 80 was measured 17x worse than nb = 10."""
    _res, seen_nb = _run([0.5] * 12)
    assert pd._MAX_BASIS_DOUBLINGS == 2, (
        "the cap moved; nb = 80 destabilised the deformation in the measurement above")
    assert max(seen_nb) <= 10 * 2 ** pd._MAX_BASIS_DOUBLINGS


def test_the_enlarged_basis_does_not_leak_into_the_caller_s_settings():
    """One temperature's basis growth must not follow the solver to the next one.

    `fullTunneling` copies `deformation_init_params`, so growth is local to a single solve.
    If it ever stopped copying, every later temperature would inherit the enlarged basis and
    the run would silently drift.
    """
    shared = {"nb": 10}
    _res, seen_nb = _run([0.02] * 6, init_params=shared)
    assert shared["nb"] == 10, (
        f"the caller's dict was mutated to nb = {shared['nb']}, so the basis would leak "
        "into every subsequent action solve")
    assert max(seen_nb) > 10, "the test did not actually trigger any growth"
