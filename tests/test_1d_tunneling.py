import unittest

import models.TL_conformal_dark_u1 as potential
from transitionlistener.phases import Phases
from transitionlistener import pathDeformation, tunneling1D

class test1DPotential(unittest.TestCase):
    """Test the unittests."""

    def test_phase_tracing(self):
        pot = potential.specific_potential({"g": 0.7, "y": 0.24, "v_GeV": 1000}, verbose=True)
        phases = Phases(pot, verbose=True)

        # Should find 2 phases
        nphases = len(phases.keys())
        self.assertTrue(nphases == 2)

        # Finds correct minima? Near zero field the zero-temperature potential
        # rises only like phi^4 log(phi^2), so the tracer places the symmetric
        # minimum only to about 1e-5 of the vev (13 MeV here, since v2.0.0).
        self.assertAlmostEqual(phases[0].X[0][0], 1000, delta=1e-3)
        self.assertAlmostEqual(phases[1].X[0][0], 0, delta=5e-2)


    def test_action_calculation(self):
        pot = potential.specific_potential({"g": 0.7, "y": 0.24, "v_GeV": 1000}, verbose=True)
        T = 15

        def V(x, T):  return pot.Vtot(x,T)
        def dV(x, T):  return pot.gradV(x,T)

        def V_(x, T=T, V=V): return V(x,T)
        def dV_(x, T=T, dV=dV): return dV(x,T)
        x0 = [0.0]
        x1 = [1000]
        fullTunneling_params = {}
        tobj = pathDeformation.fullTunneling(
                [x1,x0], V_, dV_, callback_data=T,
                **fullTunneling_params)

        # 2388.130073975295 before the path extension searched for the nearest minimum
        # instead of starting from a fixed step, and 2388.130062021543 after it: the eighth
        # significant figure, the extension landing a little differently on a path whose ends
        # are already minima.
        #
        # 2388.030513420128 since `epsabs` became a fraction of the bounce's own field
        # excursion rather than of the gap between the minima. This call takes `findProfile`'s
        # signature defaults, `phitol = xtol = 1e-4`, which no TransitionListener run uses --
        # the configured value is 1e-10. At 1e-4 the bisection's own window is what decides, and
        # this is where making it relative to the `x` it holds moves the answer most, because at
        # that tolerance neither rule has converged: `x` at the accepted trial is 0.0857, so the
        # relative window is a factor 12 the tighter and the loop takes 16 bisection trials
        # against the absolute rule's 13. The action then moves by 0.024 per cent, against
        # 3.8e-08 at the configured 1e-10, where the counts are 36 against 33. The three values
        # at 1e-4 are 2388.1301 with an absolute window and a gap-scaled epsabs, 2388.0305 with
        # an absolute window and a bounce-scaled epsabs, and 2388.6013 here, against a converged
        # 2388.1790. Looser is not closer: at the model's own 10000 spline samples the distances
        # from the converged 2384.3235 are 0.4248, 0.0390 and 0.6498. At the configured 1e-10
        # this code gives 2388.1790354 on an AMD EPYC 7453, against 2388.1789449 with an
        # absolute bisection window and 2388.1788707 with an absolute bisection window and a
        # gap-scaled epsabs. The test below separates all three: the absolute bisection window
        # by the number of bisection trials, which is the same integer on both processors, and
        # the gap-scaled epsabs by a window of 4e-05 on the action, wide enough to hold both
        # processors and narrow enough to exclude that rule.
        self.assertAlmostEqual(tobj.action, 2388.601280038287, places=5)

    def test_the_action_converges_as_the_tolerance_is_tightened(self):
        """What the number above should have been pinned against.

        A single action at one tolerance says nothing about whether it is right. The bounce has
        a value the solver is approaching, and the test of the solver is that tightening its
        tolerances walks towards that value and stays there. The Euclidean action S of this
        configuration, dimensionless in the model's internal units, moves by 1.9e-03 between
        phitol = xtol = 1e-8 and 1e-10 and then by at most 1.4e-04 over the two further decades
        of tolerance to 1e-14, so the walk has arrived by 1e-10.

        The convergence is asserted as a ratio of two measured moves in S rather than as a
        window around one value of S, and the shooting rule is asserted through the integer
        number of profile integrations the bisection runs. Neither quantity depends on the
        processor; S itself does. At phitol = xtol = 1e-10 the same code and the same library
        build give S = 2388.1790354 on an AMD EPYC 7453 and S = 2388.1789926 on an AMD EPYC
        9454, 4.27e-05 apart.
        """
        pot = potential.specific_potential({"g": 0.7, "y": 0.24, "v_GeV": 1000}, verbose=False)
        T = 15
        # `integrateProfile` runs once per bisection trial, so counting its calls counts the
        # trials. The count is taken by wrapping the method on the class, because
        # `fullTunneling` builds the `SingleFieldInstanton` itself and no instance is reachable
        # from here. The original is put back whatever the test does.
        integrations = {"n": 0}
        unwrapped = tunneling1D.SingleFieldInstanton.integrateProfile

        def counting_integrateProfile(instanton, *args, **kwargs):
            integrations["n"] += 1
            return unwrapped(instanton, *args, **kwargs)

        tunneling1D.SingleFieldInstanton.integrateProfile = counting_integrateProfile
        self.addCleanup(
            setattr, tunneling1D.SingleFieldInstanton, "integrateProfile", unwrapped)

        actions = {}
        trials = {}
        for tol in (1e-8, 1e-10, 1e-12, 1e-14):
            integrations["n"] = 0
            tobj = pathDeformation.fullTunneling(
                [[1000], [0.0]], lambda x: pot.Vtot(x, T), lambda x: pot.gradV(x, T),
                callback_data=T,
                tunneling_findProfile_params={"phitol": tol, "xtol": tol})
            actions[tol] = float(tobj.action)
            trials[tol] = integrations["n"]
        settled = (max(actions[t] for t in (1e-10, 1e-12, 1e-14))
                   - min(actions[t] for t in (1e-10, 1e-12, 1e-14)))
        approach = abs(actions[1e-10] - actions[1e-8])

        # The three tightest tolerances have to agree with the value this configuration settles
        # on. 1e-8 is excluded because it has not arrived: it sits 1.9e-03 below, which is
        # within 7 per cent of this window and would make the window the thing under test.
        for tol in (1e-10, 1e-12, 1e-14):
            self.assertAlmostEqual(
                actions[tol], 2388.179, delta=2e-3,
                msg=f"at phitol = xtol = {tol:g} the action is {actions[tol]}, which is not "
                    f"the value this configuration converges to")
        self.assertLess(
            abs(actions[1e-14] - actions[1e-12]), 1e-3,
            f"the action is still moving between 1e-12 and 1e-14: {actions}")

        # The convergence statement: the spread of S over the three tightest tolerances is a
        # small fraction of the move that brought S there from 1e-8. Written as a ratio of two
        # measured moves, the statement survives a change of processor, which an absolute
        # window on S does not. Measured on the shipped rule: the spread is 8.23e-05 on an AMD
        # EPYC 7453 and 1.40e-04 on an AMD EPYC 9454 against an approach of 1.889e-03 and
        # 1.862e-03, so the ratio is 22.9 and 13.3 and the factor 5 below holds with a margin
        # of 2.7 on the worse of the two.
        #
        # What this assertion cannot see is which shooting rule is in force, because every rule
        # converges to the same bounce: the three rules measured below give ratios of 9.9 to
        # 83.3, all of them above 5. The two assertions after it carry that part.
        self.assertLess(
            settled, approach / 5,
            f"the action has not settled: it still spreads by {settled:.3e} over the three "
            f"tightest tolerances, against {approach:.3e} between 1e-8 and 1e-10; {actions}")

        # Which rule stops the bisection is visible in the number of profile integrations, and
        # that number is an integer the floating-point grid does not move: 30, 36, 43 and 49 at
        # phitol = xtol = 1e-8, 1e-10, 1e-12 and 1e-14 on both processors. With the window of
        # the bisection put back to an absolute `xtol` rather than a fraction of the `x` the
        # bracket holds, the counts are 26, 33, 39 and 46, again on both processors.
        #
        # This is the assertion that separates the shipped rule from the one it replaced, and no
        # window on the action can do it. The shipped rule gives S = 2388.1789926 to
        # 2388.1790354 across the two processors and the absolute window gives 2388.1789449 to
        # 2388.1790097; the two ranges overlap, so no window on S accepts the first everywhere
        # and rejects the second everywhere.
        self.assertEqual(
            trials[1e-10], 36,
            f"the bisection ran {trials[1e-10]} profile integrations at the configured "
            f"phitol = xtol = 1e-10, not the 36 this stopping rule runs; an absolute xtol "
            f"window runs 33. Counts at 1e-8, 1e-10, 1e-12, 1e-14: {trials}")

        # The scale `epsabs` is measured against is the one thing left, and the integration
        # count is blind to it: with `epsabs` a fraction of the gap between the minima instead
        # of the bounce's own field excursion the counts are unchanged at 30, 36, 43 and 49.
        # That rule gives S = 2388.1788728 on an AMD EPYC 7453 and 2388.1788699 on an AMD EPYC
        # 9454, 1.41e-04 and 1.44e-04 below the centre of the window asserted below, so a
        # window of 4e-05 rejects that rule on both processors with a margin of 3.5. The centre is the midpoint of the two measurements of the
        # shipped rule and the window is 1.9 times their half-spread of 2.14e-05.
        #
        # The scope of this window is that one rule. It does not gate the stopping rule of the
        # bisection, for the reason given above, and the nearest rule it does reject sits
        # 1.41e-04 away, so widening it beyond about 1.3e-04 would leave it rejecting nothing.
        self.assertAlmostEqual(
            actions[1e-10], 2388.1790140, delta=4e-5,
            msg=f"at the configured phitol = xtol = 1e-10 the action is {actions[1e-10]}, "
                f"which is not what this code gives; a gap-scaled epsabs gives 2388.1788728 "
                f"and a gap-scaled epsabs with an absolute bisection window 2388.1788707")


if __name__ == "__main__":
    unittest.main()
