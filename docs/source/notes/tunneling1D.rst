.. _notes-tunneling1D:

``tunneling1D.py``
==================

Reasoning and measurements behind the implementation. These passages were
in the docstrings of ``src/transitionlistener/tunneling1D.py``, which now carry
the summary and point here.

.. _note-tunneling1-d-bracket-spread:

``_bracketSpread``
------------------

A branch whose action is not finite has not resolved the bounce either, and a relative spread cannot say so: `inf - inf` and `inf / inf` are both `nan`, and `nan > tol` is false, so the caller would accept the sample on its worst input. Non-finite is reported as an infinite spread, which the caller refuses.

A branch whose action is not positive is refused the same way, and for the same reason: agreement is not the only thing that makes a bounce resolved. The Euclidean action of a bounce is positive, so a branch returning zero or less has not found one, and two such branches agree perfectly with each other. Two equal negative actions give a spread of zero and would pass; `bounceAction`'s caller then replaces a negative action with zero and labels the point a second-order transition, so a solver failure would be delivered as a physical result. The same measurement that set the tolerance saw it: at lambda3 = 5.816407666, T = 9.8546 GeV one of the two branches returns a negative action.

.. _note-tunneling1-d-single-field-instanton-find-profile-comment:

``SingleFieldInstanton.findProfile``, implementation comment
------------------------------------------------------------

`integrateProfile` spends `epsabs` twice, and the second use is the one that decides. Besides controlling the Runge-Kutta step it is the radius in `abs(y - [phi_metaMin, 0]) < 3*epsabs` inside which a profile counts as having reached the false vacuum. Tied to the gap that radius was a fixed absolute distance while the bounce shrank with the temperature, so `3 phitol |delta_phi| / excursion` grew from 5.9e-05 at T_int = 1e-3 to 0.069 at 1e-6 and 0.63 at 1e-7, where a trial counted as arrived once it came within two thirds of the whole bounce of the false vacuum and the action came back 0 against the 102.93 this code gives. Separating the two uses shows which one matters: tightening only the step control gives 91.10 at T_int = 1e-6 and still 0 at 1e-7, tightening only the acceptance radius gives 96.93 and 636.3, and both together 103.20 and 102.93, against a converged 102.46 and 92.42. Tied to the bounce, `3*epsabs/excursion` is `3*phitol` at every temperature.

Where the bounce covers a sizeable part of the gap the two rules are expected to agree, and the two configurations measured so far do: on the g = 0.7, y = 0.24 configuration at T = 15 that `tests/test_1d_tunneling.py` uses they differ by 3.1e-08 relative at the configured `phitol = 1e-10`, and on `examples/example_point.yaml` the observables are identical to five figures. Away from the supercooled line that is the whole of the evidence.


.. _note-tunneling1-d-single-field-instanton-find-profile-comment-1:

``SingleFieldInstanton.findProfile``, implementation comment (2)
----------------------------------------------------------------

A wrong sign is what the path's own potential produces near the true vacuum. `SplinePath.dV` differentiates a cubic spline through sampled `V` whenever `V_spline_samples` is set, and that spline ripples. Measured on the 2HDM at lambda3 = 5.8191375291 with `TL_2HDM_BSMPT_highacc`, V samples uniformly spaced at 0.24627 in path length: the ripple in `V` is 2 to 3 over a half spacing, so the derivative carries an absolute error near 20, against an analytic `grad V . dpath/dx` of +0.0118 at the endpoint itself. The spline returns -23.5875 there at T = 60 GeV, -196.002 at 58 GeV and +7.79409 at 55 GeV: the sign is set by where the samples fall. Because the true derivative vanishes at a minimum, no sampling density removes this; the relative error always diverges as the endpoint is approached.

`dV_from_absMin` covers the region with a Taylor form, but only out to `phi_eps`, while the spline's derivative becomes usable only once `d2V delta_phi0` clears the error floor, five times further out at that benchmark. In between the returned `dV` is negative, and a bisection trial landing there used to abort the whole solve, which `_find_profile` reports as an infinite action. The bounce itself is unaffected: it converges at `x` near 3.8, `delta_phi0` near 5.67, where the spline's `dV` agrees with the analytic one to 1e-5. Only stray trials visit the unsound region, and they fell in a band of temperatures with sharp non-monotonic edges, because whether a trial lands inside it is an accident of the bisection path.

So an unusable slope is a verdict on this `x`, not on the solve. It is rejected the way an over- or undershoot is, and the bisection carries on. Which bracket end it replaces follows from where `phi0` sits: nearer `phi_absMin` the trial bounds `x` from above, nearer `phi_metaMin` from below.

What has to be positive is the slope that drives the field *away* from `phi_absMin`, which is `dV` times the direction `delta_phi0` points in, not `dV` itself. The two agree only while `phi_metaMin > phi_absMin`, which is the orientation every TransitionListener run takes, because `fullTunneling` measures the field coordinate as path length from the true vacuum. They differ for a descending one, and testing `dV` alone refuses every trial there: on the quartic `0.25 phi^4 - (1+d)/3 phi^3 + d/2 phi^2` with d = 0.1, whose minima are at 0 and 1, `phi_absMin = 1, phi_metaMin = 0` raises on the first trial while the same potential mirrored through `phi -> 1 - phi` solves. Taking the direction into account makes the two orientations agree, and leaves the ascending one untouched.


.. _note-tunneling1-d-single-field-instanton-find-profile-comment-2:

``SingleFieldInstanton.findProfile``, implementation comment (3)
----------------------------------------------------------------

Where they do not, the bounce is not resolved at this temperature, and the profile that would be returned is whichever trial happened to come last. Measured on the 2HDM at lambda3 = 5.816407666 with `TL_2HDM_BSMPT_highacc`: at T = 9.8546 GeV the bisection runs 296 integrations whose last six share one `x` while the verdict alternates, and the two branches give +3743.07 at a bubble radius of 1.206 and -501.35 at 3.214. A negative Euclidean action is impossible, so one of them is certainly wrong, and which one is returned changes between temperatures 2e-04 GeV apart. Reporting the failure lets the controller drop the sample, which is what it already does with an action it cannot compute; returning either number silently puts an `S_3/T` of 73 where the curve is near 440. A solve that did not converge is unresolved until two integrated branches have been compared and found to agree. Starting the spread at zero made the opposite assumption: a bisection that ended with only one side ever integrated, so that there was no pair to compare, left the spread at zero and `fullTunneling` took the last profile as resolved on the strength of a test that had not been run.

.. _note-tunneling1-d-single-field-instanton-find-profile-comment-3:

``SingleFieldInstanton.findProfile``, implementation comment (4)
----------------------------------------------------------------

Check if we've reached xtol, as a fraction of the `x` the bracket holds rather than as an absolute window. `x` is the fraction of the gap between the minima that the bounce covers, `|phi(0) - phi_metaMin| / |delta_phi| = |exp(-x) - 1| ~ x`, so it shrinks with the supercooling while an absolute `xtol` does not: on the shipped conformal dark U(1) at g = 0.46, y = 0.01, v_GeV = 0.14, `xtol/x` runs from 1.6e-07 at T_int = 0.1 to 2.3e-02 at 1e-6. The cost shows in the number of bisection trials the absolute rule allows: on that model at the configured `xtol = 1e-10`, with this line put back to `(xmax-xmin) < xtol`, it is 25, 22, 19, 15, 12 and 8 at T_int = 1e-1 down to 1e-6, falling as the transition cools because an absolute `xtol` is already wide against the bracket once `x` is small. With the window a fraction of `x` it is 36 at every one of those temperatures. `xmin` enters so the test is defined on the first iteration, before `x` has been bracketed from below; note that `xmax` starts at infinity, so no stopping test fires until an overshoot has bracketed `x` from above.

The scale is capped at one so the window is never wider than the absolute `xtol` it replaces. Without the cap this would loosen the bisection wherever the bracket sits above `x` = 1, which is the thick-walled end and not a corner case: the 2HDM benchmark of `TL_2HDM_BSMPT_highacc` converges at `x` = 2.27 to 2.45, and the shipped conformal dark U(1) at g = 0.7, y = 0.24 passes `x` = 1 at T = 97.5 and reaches 2.02 at T = 135. Without the cap the bisection stopped 1.01 to 1.16 times `xtol` apart there, looser than the absolute rule it replaces. `findProfile` is a general solver and nothing here constrains `x`, so the cap keeps the change a tightening where the bounce is small, the absolute rule unchanged where it is not, and a loosening nowhere.

The bracket cannot be subdivided below the spacing of the floating-point grid. Once the midpoint lands on an endpoint every further trial repeats the same `x`, and no tolerance test can fire, so the loop has to stop on that rather than on `xtol`. It matters because the window is relative: the threshold `xtol * x` and the spacing `ulp(x)` both scale with `x`, so the stall sets in below `xtol` = 2.2e-16 at every `x`, where the absolute rule only stalled below `ulp(x)` itself and so needed an `xtol` under 5.2e-26 on the supercooled line. Measured with `xtol` = 1e-16: without this test the shipped conformal dark U(1) at g = 0.46, y = 0.01 and at g = 0.7, y = 0.24 both pass 400 trials with the last six `x` identical, and with it they stop.
