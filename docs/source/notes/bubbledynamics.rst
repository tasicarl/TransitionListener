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

* ``betaH_S3_fit_points`` nearest samples are used; * at least ``betaH_S3_fit_min_per_side`` of them must lie on each side of ``T`` and none further than ``betaH_S3_fit_max_rel_span`` in ``|T'/T - 1|``, otherwise five new actions are computed at T (1 + k step), k = -2..2, with step ``betaH_S3_fallback_rel_step``, and the same fit is applied; * the fit is repeated with the ``betaH_S3_fit_check_points`` nearest samples, and a relative difference above ``betaH_S3_fit_rel_tol``, measured against the larger of the two, is reported through ``diagnostics["fit_unstable"]``.

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

