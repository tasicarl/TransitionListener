.. _notes-bubbledynamics:

``bubbledynamics.py``
=====================

Reasoning and measurements behind the implementation. These passages were
in the docstrings of ``src/transitionlistener/bubbledynamics.py``, which now carry
the summary and point here.

.. _note-bubbledynamics-entropy-density:

``entropy_density``
-------------------

Only ratios and logarithmic derivatives of this quantity are used, so the constant ``2 pi^2/45`` is dropped from the counted form and the two schemes are not on a common absolute normalisation.

Both describe the same plasma: the fields of the potential and the coupled radiation bath, without the decoupled one. They differ in how they treat the fields. ``"dof_table"`` counts them as free particles at their zero-temperature masses and adds the tabulated entropy degrees of freedom of the coupled bath. ``"eff_potential"`` takes ``-dV/dT``, which gives the fields their thermal masses and the Arnold-Espinosa daisy term, and which carries the tabulated bath with its perturbative corrections as well, since the field-independent part of the potential is built from the same tables. What it does not have is those corrections for the fields of the potential themselves. Neither route contains the other; see the ``entropy_definition`` setting.

.. _note-bubbledynamics-double-integral-subgrid:

``_double_integral_subgrid``
----------------------------

The source ``Gamma a^3 / (3 c_s^2 H T)`` is resolved to a change of at most ``_SUBGRID_MAX_LOG_STEP`` in its logarithm per sub-step, wherever it is within ``exp(_SUBGRID_NEGLIGIBLE)`` of its peak. Between the support points ``S/T``, ``ln H``, ``3 c_s^2`` and ``ln a`` are interpolated in ``u = ln T`` by a not-a-knot cubic spline through the support points of that region and one more on each side; the support points themselves keep their values. A rate with a Gaussian turnover, ``ln Gamma`` quadratic in time, is then reproduced to 0.1 % across gaps of 0.7 e-folds, where PCHIP is off by a few per cent. The spline spans the support points from one before the first refined interval to one after the last, so points outside that range, such as a far hot shoulder left whole because the source there is negligible, contribute no nodes. Intervals with a non-finite end, and grids that are fine already, are returned unchanged.

.. _note-bubbledynamics-perc-integral-ode:

``percIntegralODE``
-------------------

With ``i_target`` the solver additionally records the temperature at which the percolation integral reaches that value, and the return value becomes the pair ``(I_values, crossing_temperature)``, where the crossing is ``None`` if the target is not reached. Taking the crossing from the integrator avoids reading it back off an interpolation of the sampled ``I`` values.

With ``sound_speed_sq`` and ``scale_factor`` this solves the generalized EOS form using ``gamma = Gamma a^3 / (H T 3 c_s^2)`` and ``nu = 1 / (H T 3 c_s^2 a)``. Without those arrays it falls back to the historical bag-limit relation, where ``c_s^2 = 1/3`` and ``a(T)`` cancels to the simpler transport factor ``1/H``.

.. _note-bubbledynamics-bracket-sign:

``_bracket_sign``
-----------------

The two infinities are not symmetric, and treating them as if they were puts back the defect this walk exists to remove.

`-inf` is a definite answer. The criterion returns it for `T >= Tmax` and for `T + dT >= Tmax` before computing any action, and again when an action is exactly zero at either sample, and in each case it means "above the root". Discarding it skips the sign test exactly where the criterion is most emphatic: with the hot root inside the first step, at 110 against a `Tmax` of 120, requiring both ends finite returns 9.9977, the artefact at the cold end of the band, and with a degenerate `-inf` one step above a root at 60 it returns 9.99774. Counting it by its sign returns the root in both.

`+inf` carries no information. The criterion returns it at `T <= Tmin`, which is construction rather than measurement, and wherever an action is not computable. Either way, bracketing against it manufactures a root at the cold end of whatever interval it is handed, which is how the estimate came to return 1.96582e-12 GeV. A stretch with no computable action is bounded by `_APPROX_PERCOLATION_MAX_BLIND_STEPS` instead.

`brentq` tolerates an infinite value at a bracket end, measured for roots at 110, 118 and 119.9 of a `Tmax` of 120, and the whole-interval call this walk replaces handed it `Tmax` with that same `-inf` at every point TransitionListener has ever run.

.. _note-bubbledynamics-integrate-broken-temperature:

``integrate_broken_temperature``
--------------------------------

The step-3 profile obtains the broken-phase temperature by stepping from one support point to the next: an entropy-conserving update followed by an energy correction for the volume fraction converted during the step. Those steps are a first-order discretisation of

.. math::
    \frac{dT_b}{dT} = \frac{3\,S(T_b)}{T\,S'(T_b)}
    + \frac{P'(T)}{P(T)}\,\frac{e_s(T) - E(T_b)}{E'(T_b)} ,

with :math:`S(x) = [h_{\rm ds}(x) + h_{\rm SM}(x)]\,x^3` proportional to the entropy density of the broken phase and :math:`E` its energy density. Reading the reheating temperature off that tabulated trajectory makes it depend on where the support points happen to sit, which shows up as point-to-point scatter along smooth parameter scans. Solving the equation with an error-controlled integrator makes the result depend on a tolerance instead.

The true-vacuum fraction enters through the source term. Its derivative is taken on a spline of :math:`\ln I`, with :math:`I = -\ln(1-P)` the percolation integral, which is smooth and monotone, rather than on a spline of :math:`P`, which sweeps from 0 to 1 and whose derivative depends strongly on the node placement. The integration starts where the true-vacuum fraction first reaches ``p_seed``, located in the same variable, with the instantaneous-reheating condition as initial value.

Returns the broken-phase temperature at ``T_target``, or ``None`` if the integration could not be carried out.

.. _note-bubbledynamics-recompute-actions-tight:

``_recompute_actions_tight``
----------------------------

``progress`` is the caller's record of what happened: the cache entries removed, under ``dropped``, so that the caller can put the cache back as it was, and the temperatures actually recomputed, under ``done``. Both matter where this raises part of the way through, when the caller has to restore what was taken and report what was really attempted rather than what was planned.

``calcAction`` hands back whatever is already in ``outdict`` for a temperature, so an action computed at the loose deformation would be returned unchanged and the recomputation would be silent about it. The percolation temperature is always in there. Every entry the stencil lands on is therefore dropped first, as the rate-jitter rescue drops the one it replaces, and kept for the record.

.. _note-bubbledynamics-calc-beta-h-s3:

``calc_betaH_S3``
-----------------

beta/H = T d(S3/T)/dT at ``T``, estimated by a local least-squares quadratic through the action samples of the percolation support rather than from the slope of an interpolating spline. The adaptive support places samples as close as 1e-6 T apart; an interpolant forced exactly through them converts an irregular 1e-2 error in S3/T (path-deformation noise) into errors of hundreds in beta/H, while a fit over the nearest samples averages it out.

The sample selection is controlled by ``percolationConf``:

* ``betaH_S3_fit_points`` nearest samples are used.
* At least ``betaH_S3_fit_min_per_side`` of them must lie on each side of ``T`` and none
  further than ``betaH_S3_fit_max_rel_span`` in ``|T'/T - 1|``, otherwise five new actions
  are computed at T (1 + k step), k = -2..2, with step ``betaH_S3_fallback_rel_step``, and
  the same fit is applied.
* The fit is repeated with the ``betaH_S3_fit_check_points`` nearest samples, and a relative
  difference above ``betaH_S3_fit_rel_tol``, measured against the larger of the two, is
  reported through ``diagnostics["fit_unstable"]``.

Both sample counts are clamped: a quadratic needs three samples, and the check fit uses at most as many as the fit itself, in which case it is skipped.

See the CAUTION note in ``PercolationConf``: the defaults were validated on the 2HDM BSMPT benchmark only.

.. _note-bubbledynamics-false-vacuum-volume-growth-rate:

``falseVacuumVolumeGrowthRate``
-------------------------------

The false vacuum occupies a physical volume :math:`\mathcal{V}_{\rm false} \propto a^3 (1 - P_{\rm true}) = a^3 e^{-I}`, which has to shrink for the transition to complete rather than to inflate forever. This requirement goes back to Turner, Weinberg and Widrow, Phys. Rev. D 46 (1992) 2384; the temperature form used here is eq. (2.26) of Ellis, Lewicki and No, arXiv:1809.08242, and we refer to it as the Lewicki criterion,

.. math::
    \frac{1}{\mathcal{V}_{\rm false}}
    \frac{{\rm d}\mathcal{V}_{\rm false}}{{\rm d}t}
    = H\left(3 + T\frac{{\rm d}I}{{\rm d}T}\right) < 0 \,.

That form assumes the radiation relation :math:`{\rm d}T/{\rm d}t = -HT`. TransitionListener integrates the percolation history with the generalized relation :math:`{\rm d}T/{\rm d}t = -3 c_{\rm s}^2 H T`, for which

.. math::
    \frac{1}{\mathcal{V}_{\rm false}}
    \frac{{\rm d}\mathcal{V}_{\rm false}}{{\rm d}t}
    = 3H\left(1 + c_{\rm s}^2 T \frac{{\rm d}I}{{\rm d}T}\right) \,,

and the returned quantity is the bracket. With :math:`c_{\rm s}^2 = 1/3` it reduces to :math:`(3 + T\,{\rm d}I/{\rm d}T)/3`, i.e. to the published criterion divided by three, so the sign is unchanged.

The derivative is taken on a monotonic spline of :math:`\ln I` rather than of :math:`P`, because :math:`I = -\ln(1 - P_{\rm true})` spans decades over the sampled range while :math:`P` saturates.

.. _note-bubbledynamics-record-entropy-scheme-diagnostics:

``record_entropy_scheme_diagnostics``
-------------------------------------

All three keys are set on every path. Where the comparison does not apply, because it is switched off, because the time-temperature relation reads no entropy, or because the configured history could not be built, the flag is ``False`` and both numbers are ``nan``: the keys are registered for every run, so leaving them unset would have the writer report them as unimplemented and fill all three with ``nan``, the boolean included.

A timeout reaches the caller. Any other failure of the comparison leaves the ``nan`` and must not invalidate the point, which was computed before the comparison was asked for.

.. _note-bubbledynamics-entropy-scheme-sensitivity:

``entropy_scheme_sensitivity``
------------------------------

The two schemes of ``ENTROPY_DEFINITIONS`` are two approximations to one entropy density, and neither contains the other, so the difference between them is a modelling uncertainty of the expansion history rather than a numerical error. It is reported rather than corrected, because nothing in the literature chooses between them.

Two numbers, both on the support that fixed the percolation temperature:

``cs_sq_spread``
    the median relative difference between the two schemes' ``3 c_s^2`` over the
    support. Measured over 91 points of four scan lines against the shift each point's
    mean bubble separation actually takes between the schemes, this is an estimate of
    that shift: the ratio of the two has median 1.06 and lies between 0.88 and 1.29 for
    eight points in ten, and 93% of points agree within a factor of two. The bias is the
    same on every line tested, between 1.00 and 1.17.
``scale_factor_shift``
    the gap between the two schemes in ``|delta ln a|`` across the support, not the
    expansion itself. Since ``d ln a/d ln T = -1/(3 c_s^2)`` and
    ``ln a = -(1/3) ln s`` up to a constant, this is the difference of the two schemes'
    logarithmic entropy ratios over three, and it is the first-order estimate of the
    relative shift it would produce in the mean bubble separation.

Pass ``configured_history`` as the ``(logEntropyInt, coolingInt)`` pair already built for the configured scheme, and only the other scheme is evaluated: one further history rather than two, and the comparison is then made against the very history the observables came from. One history is three `entropy_density` evaluations per support point, at the temperature and at both endpoints of the derivative stencil, so handing the configured one over halves the cost of the comparison.

.. _note-bubbledynamics-log-gamma-source-floor-comment:

``_LOG_GAMMA_SOURCE_FLOOR``, implementation comment
---------------------------------------------------

The standard ``percIntegral`` computes I(T) via a nested double integral at O(N^2) per evaluation (O(N^3) for a full sweep over the grid).  ``percIntegralODE`` reformulates I(T) as a 4-component ODE chain that can be integrated in a single O(N) sweep from T_hot to T_cold:

::

   A(T)  = Gamma(T) / (T^4 * H(T))          (nucleation-rate density)
   J_0'  = -A              J_0(T_hot) = 0
   J_1'  = -J_0 / H        J_1(T_hot) = 0
   J_2'  = -2 J_1 / H      J_2(T_hot) = 0
   J_3'  = -3 J_2 / H      J_3(T_hot) = 0

::

   I(T)  = (4 pi / 3) * v_w^3 * J_3(T)
   P(T)  = 1 - exp(-I(T))

Here ' = d/dT (with T *decreasing* toward the cold boundary).


.. _note-bubbledynamics-calc-approx-percolation-comment:

``calcApproxPercolation``, implementation comment
-------------------------------------------------

Find the *hottest* temperature at which the criterion is met, by walking down from `Tmax` until it changes sign, and only then handing that sub-bracket to `brentq`.

Handing `brentq` the whole interval does not work, because the interval carries more than one root and `brentq` assumes one. The criterion is positive below the percolation point and negative above it, but as `T` falls the rate vanishes, so it turns negative again at the cold end, and `T <= Tmin` returns `+inf` by construction. Percolation is the first temperature the universe meets on the way down, so the hottest root is the one wanted and the others are artefacts of the bracket.

The walk is geometric at `_APPROX_PERCOLATION_SCAN_RATIO` per step, which costs two action evaluations each and reaches the root in a handful of steps because the root sits within a factor of a few of `Tmax`, so the whole estimate costs on the order of ten criterion calls rather than a walk over the interval. It stops where no bubble can nucleate at any Hubble rate, since no root below that is physical, and separately after `_APPROX_PERCOLATION_MAX_BLIND_STEPS` consecutive steps whose action could not be computed, which is the case the first stop cannot see because it has no `S_3/T` to read.


.. _note-bubbledynamics-jitter-dead-rate-decades-comment:

``_JITTER_DEAD_RATE_DECADES``, implementation comment
-----------------------------------------------------

Both ends of the window are constrained, and by different things. Open it much wider and the gate goes back to rejecting the dead tail it was written to ignore, since every false rejection seen came from samples forty decades or more below the peak. Close it much further and the window starves the fit, which needs five active samples for its quadratic: by thirty decades a profile can already have only two support points inside the window, and narrowing it moves more profiles into the "unassessable" state, which measures nothing. 30 sits between the two, with 1.35x margin to the shallowest observed false positive.

Note what this does *not* claim: that a sample below the floor cannot matter at all. The grid builders' comment records contributing intervals -- worth more than a millionth of the percolation integral -- as deep as 448 decades below a profile's own peak, with a median of 34, so depth alone does not bound a sample's weight. What the floor says is that below it the gate cannot tell a bad solve from its own inability to fit, so it reports the exclusion in `n_dead_rate_excluded` rather than a verdict.


.. _note-bubbledynamics-try-action-jitter-tunneltight-rescue-comment:

``_try_action_jitter_tunneltight_rescue``, implementation comment
-----------------------------------------------------------------

What the discard does and does not do. The non-finite action is dropped by the action interpolants built from the returned `Sr`, so the rate feature it carried no longer reaches `R_*`, `betaH_RH`, `betaH_S3`, the spline nucleation-temperature estimate or the jitter diagnostic itself. What it does not do is re-derive the profile, and the percolation integral has already run with the sample in it: this is called from `_raise_if_action_rate_jitter_unresolved`, which runs after `_refine_percolation_temperature` and `_finalize_percolation_state`, so `Pr`, `Tb`, `Tperc` and the support bank still describe the profile that included it, and the Delta P and hot-head acceptance gates are not run again on what remains. The returned `Sr` and `Pr` therefore describe slightly different supports.

That is tolerated rather than hidden, because both alternatives measured worse. Re-deriving the profile from the reduced support and re-running the gates leaves the coupling's inverse duration far from the trend its neighbours set, and refusing the point instead costs every coupling that discards at all.

The couplings that reached the discard were sick for a reason outside this file: the traced symmetric phase was interpolated with an overshooting cubic, so the bounce tunnelled from a displaced point and returned too small an action, and it was that step which tripped the gate above. `PhaseInfo._build_interpolator` now interpolates the trace with a shape-preserving `PchipInterpolator`, which removes that cause. The discard stays, because nothing stops another model from tripping the gate for a reason of its own, and the state it leaves is still the price; whether it fires at a given point is not visible in the output, because `action_jitter_rescue_discarded_temperatures` has no consumer.

