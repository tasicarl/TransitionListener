.. _notes-helper-functions:

``helper_functions.py``
=======================

Reasoning and measurements behind the implementation. These passages were
in the docstrings of ``src/transitionlistener/helper_functions.py``, which now carry
the summary and point here.

.. _note-helper-functions-step-error-estimate:

``_step_error_estimate``
------------------------

The smaller of the absolute and fractional comparisons for each component, and the largest of those over the components. `rkqs` accepts the step where this is below 1.

The same number as `np.nan_to_num(np.max(np.min([a, b], axis=0)))`, computed without building a list of two arrays for `np.min` to stack. The bounce solver asks for it once per Runge-Kutta step, 78658 times for six bounces of the real two Higgs doublet model benchmark, which is why the arithmetic is written out.

`np.nan_to_num` is spelled out rather than called because it is a general-purpose function working on a single number here: it sends a `nan` to zero, so that a step whose error cannot be judged is accepted, which is the behaviour the solver has always had, and an infinity to the largest value the reduction's own dtype can hold. That dtype is read from the reduction rather than fixed at `float64`, because that is what `np.nan_to_num` does: a `float32` `yerr` gives 3.4028235e+38 and not 1.7976931e+308.

.. _note-helper-functions-temperature-derivative-step:

``temperatureDerivativeStep``
-----------------------------

The stencils in :meth:`generic_potential.dVdT` and :meth:`generic_potential.d2VdT2` difference the *full* effective potential, which in the broken phase can carry a large temperature-independent vacuum piece. Cancelling that offset leaves a round-off error of order :math:`\epsilon |V| / \Delta T^2` in the second derivative, while the five-point stencils truncate at order :math:`\Delta T^4`. Balancing the two gives

.. math::
    \frac{\Delta T}{T} \simeq
    \left(\epsilon \frac{|V|}{\rho_{\rm rad}}\right)^{1/6} \,,

for the five-point stencils. A two-point central difference of a *first* derivative balances a round-off of order :math:`\epsilon |V| / \Delta T` against a truncation of order :math:`\Delta T^2`, which gives the cube root of the same ratio instead; pass ``two_point=True`` for those. The cube root is much smaller than the sixth root whenever the ratio is large, so using the five-point step for a two-point derivative costs truncation error: 2e-4 on the potential difference of the conformal benchmark.

where the radiation energy density stands in for the thermal scale that sets the size of the temperature derivatives. Both are taken for the sector the derivative itself uses, selected by ``include_decoupled``. A radiation-dominated phase has :math:`V = -p = -\rho/3`, so the ratio cannot fall much below one third and the step stays above :math:`(\epsilon/3)^{1/6} \simeq 2\times10^{-3}`; the lower clip is a guard, not a working bound. The two regimes this has to serve differ by orders of magnitude: for the conformal dark U(1)' benchmark :math:`|V| \sim 3\times10^8` against a thermal scale of order one, so the step has to be about a per cent of ``T`` or the sound speed in the broken phase picks up a 0.1 % scatter and a 0.3 % bias; for the abelian dark Higgs the vacuum offset is comparable to the thermal scale, no cancellation occurs, and a per-cent step already costs 0.4 % of truncation error in ``alpha``. The rule above lands on the accurate plateau in both, and is clipped to :math:`[10^{-4}, 3\times10^{-2}]`.

Without ``X`` the field-independent fallback of ``1e-2`` is used.

.. _note-helper-functions-thermal-derivative-step:

``thermalDerivativeStep``
-------------------------

:func:`temperatureDerivativeStep` cannot serve :meth:`generic_potential.dV_thermal_dT` and :meth:`generic_potential.d2V_thermal_dT2`. It sizes the step from :math:`|V_{\rm tot}|`, whose vacuum offset is exactly what those two do not carry, and its sixth-root balance is derived for five-point stencils where they are two- and three-point ones. Applied to them it saturates its own :math:`3\times10^{-2}` ceiling below :math:`T/v = 10^{-3}`, which costs accuracy in the fourth digit of the sound speed.

What binds for the ratio :math:`c_s^2 = \partial_T V_{\rm th}/(T\,\partial_T^2 V_{\rm th})` is the round-off of the second difference against the truncation of the first:

.. math::
    \frac{4\epsilon |V_{\rm th}|}{\Delta T^2\, |\partial_T^2 V_{\rm th}|}
    \;\simeq\;
    \frac{\Delta T^2 |\partial_T^3 V_{\rm th}|}{6\, |\partial_T V_{\rm th}|} \,.

The pairing is not the obvious one. The second difference is the noisier of the two, by a factor :math:`1/\Delta T`, so it sets the round-off; the fourth derivative that would set *its* truncation nearly vanishes wherever the daisy term dominates, because there :math:`V_{\rm th} \propto T^3` and :math:`n(n-1)(n-2)(n-3) = 0` at :math:`n = 3`. The third derivative does not vanish, so the first difference sets the truncation.

For :math:`V_{\rm th} \propto T^n` this gives

.. math::
    \frac{\Delta T}{T} \simeq
    \left(\frac{24\,\epsilon}{n(n-1)^2(n-2)}\right)^{1/4} \,,

and the magnitude :math:`|V_{\rm th}|` cancels out of it, because round-off and truncation both scale with it. So the step is a pure number, and sizing it on whichever contribution dominates the thermal part is the derivation of that number rather than a reason for it to vary: radiation, :math:`n = 4`, gives :math:`(\epsilon/3)^{1/4} = 9.3\times10^{-5}`, and a daisy-dominated :math:`n = 3` gives :math:`(2\epsilon)^{1/4} = 1.45\times10^{-4}`. The two differ by 50 %, the optimum is flat, and the radiation value is used for both.

Verified on the conformal dark U(1) at ``g = 0.455``, ``0.7`` and ``0.95``, at internal temperatures from :math:`10^{-1}` to :math:`10^{-9}` of the scale: ``c_s`` at this step agrees with the plateau measured over :math:`\Delta T/T \in [3\times10^{-5}, 10^{-3}]` to :math:`4\times10^{-8}` or better everywhere, and at :math:`T/v = 10^{-9}` returns ``0.70710672`` against the analytic daisy limit :math:`1/\sqrt{2} = 0.70710678`. Round-off takes over below :math:`\Delta T/T \sim 10^{-5}` and truncation above :math:`3\times10^{-3}`.

