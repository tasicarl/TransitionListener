"""fixed_step_size percolation solver.

This module keeps the compatibility path separate from the default adaptive_step_size
controller.  It intentionally reuses the shared thermodynamic helpers from
``bubbledynamics`` instead of duplicating the physics expressions here.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

import numpy as np
from scipy import interpolate, optimize

from transitionlistener import errors
from transitionlistener.bubbledynamics import (
    PercolationGrid,
    PercolationState,
    HubbleParameter,
    Tb_criterion,
    _temperature_grid,
    calcAction,
    energyDensity,
    energy_criterion_BRO,
    entropy_criterion_SYM_BRO,
    percIntegral,
)


def _initial_percolation_scan(
    grid: PercolationGrid,
    settings,
    outdict: dict,
    pot,
    phase_symmetric,
    phase_broken,
    CF: float,
    vw: float,
    verbose: bool,
) -> PercolationState:
    """Step 2: compute P(T) with a P-independent Hubble rate.

    This is the historical fixed-grid solver from main. It keeps the original
    equidistant TSYM logic and only records a little extra diagnostic state.
    """
    TSYM = grid.TSYM
    Sr = np.zeros_like(TSYM)
    Hr = np.zeros_like(TSYM)
    Pr = np.zeros_like(TSYM)
    Tb = np.zeros_like(TSYM)

    iterator = 1
    iteration_condition = True
    lowest_temperature_reached = float(TSYM[-1])

    if verbose:
        print("\nPercolation step 2: calculating Tperc assuming P = 0 in the Hubble rate")

    while iteration_condition:
        lowest_temperature_reached = min(lowest_temperature_reached, float(TSYM[-1]))
        Sr.fill(0)
        Hr.fill(0)
        Pr.fill(0)
        Tb.fill(0)
        is_Pr_converged = False

        for i, T in enumerate(TSYM):
            if not is_Pr_converged:
                Sr[i] = calcAction(pot, T, phase_symmetric, phase_broken, outdict)
                rho = energyDensity(pot, phase_symmetric, T)
                Hr[i] = HubbleParameter(rho, CF)
                Pr[i] = 1 - np.exp(-percIntegral(TSYM[0 : i + 1], Hr[0 : i + 1], Sr[0 : i + 1], vw=vw))
            else:
                Sr[i] = np.nan
                rho = energyDensity(pot, phase_symmetric, T)
                Hr[i] = HubbleParameter(rho, CF)
                Pr[i] = Pr[i - 1]

            if verbose:
                print(
                    f"Step {i+1}/{len(TSYM)}, T = {TSYM[i] * CF:2.8g} GeV, "
                    f"P_true = {Pr[i]:2.5g}, S_3 = {Sr[i] * CF:2.5g} GeV, "
                    f"S_3/T = {Sr[i] / T:2.5g}"
                )

            if i >= 1 and Pr[i] > settings.f_final and Pr[i - 1] > settings.f_final:
                if verbose and not is_Pr_converged:
                    print(
                        "WARNING: P > ",
                        settings.f_final,
                        " at T = ",
                        TSYM[i] * CF,
                        " GeV detected. Stop computation of action from here on "
                        "and only fill in the values for the Hubble rate.",
                    )
                is_Pr_converged = True

        boundary_n = min(settings.max_boundary_n, len(Pr) - 1)
        if Pr[0] <= settings.f_start and Pr[-1] >= settings.f_final:
            jump_indices = np.where((Pr[:-1] == 0) & (Pr[1:] == 1))[0]
            has_jump_0_to_1 = jump_indices.size > 0
            if has_jump_0_to_1:
                max_index = jump_indices[0]
                min_index = jump_indices[0] + 1
                tsym_jump_max = TSYM[max_index]
                tsym_jump_min = TSYM[min_index]
                if verbose:
                    print(
                        "Warning: Pr includes a 0 followed by 1, i.e. a "
                        "discontinuous jump in true vacuum fraction."
                    )
                    print(
                        f"Jump occurs between the indices {max_index} and {min_index}, "
                        f"TSYM values: {tsym_jump_min} and {tsym_jump_max}. "
                        "Set these as new boundaries."
                    )
                if tsym_jump_max / tsym_jump_min - 1 > 1e-10:
                    TSYM = np.linspace(tsym_jump_max, tsym_jump_min, len(TSYM))
                    grid.TSYM = TSYM
                    Sr = np.zeros_like(TSYM)
                    Hr = np.zeros_like(TSYM)
                    Pr = np.zeros_like(TSYM)
                    Tb = np.zeros_like(TSYM)
                    iterator += 1
                    iteration_condition = iterator < settings.maxit
                else:
                    if verbose:
                        print(
                            "WARNING: Even though there is a jump in P_true, the "
                            "temperature range is already very small. We will not "
                            "adjust it further and call the percolation temperature "
                            "calculation converged."
                        )
                    iteration_condition = False

            elif Pr[boundary_n] == 0 or Pr[-boundary_n] == 1:
                if verbose and iterator != settings.maxit:
                    print(
                        "Warning: the temperature range for the percolation "
                        "temperature computation was chosen too wide in iteration ",
                        iterator,
                        ". Decrease it!",
                    )

                if Pr[boundary_n] == 0:
                    Tmax_new = TSYM[np.where(Pr < settings.f_start)[0][-1]]
                else:
                    Tmax_new = TSYM[0]

                if Pr[-boundary_n] == 1:
                    Tmin_new = TSYM[np.where(Pr > settings.f_final)[0][0]]
                else:
                    Tmin_new = TSYM[-1]

                if Tmax_new < Tmin_new:
                    raise errors.PercolationError(
                        "The temperature range for the percolation temperature "
                        "computation was messed up. This should not happen. Please report this as a bug."
                    )
                TSYM = np.linspace(Tmax_new, Tmin_new, len(TSYM))
                grid.TSYM = TSYM
                Sr = np.zeros_like(TSYM)
                Hr = np.zeros_like(TSYM)
                Pr = np.zeros_like(TSYM)
                Tb = np.zeros_like(TSYM)
                iterator += 1
                iteration_condition = iterator < settings.maxit
            else:
                iteration_condition = False
        else:
            if verbose and iterator != settings.maxit:
                print(
                    "Warning: the temperature range for the percolation "
                    "temperature computation was chosen too narrow in iteration ",
                    iterator,
                    ". Increase it!",
                )
            if verbose and iterator == settings.maxit:
                print(
                    "Warning: the temperature range for the percolation "
                    "temperature computation was chosen too narrow. This is "
                    "the last iteration, so we will use the current "
                    "temperature range anyway."
                )
            if Pr[0] > settings.f_start:
                Tmax_new = TSYM[0] + (grid.tmax - TSYM[0]) * settings.rel_increment
            else:
                Tmax_new = TSYM[0]

            if Pr[-1] < settings.f_final:
                if iterator >= settings.maxit - 2:
                    Tmin_new = grid.tmin
                else:
                    Tmin_new = TSYM[-1] - (TSYM[-1] - grid.tmin) * settings.rel_increment
            else:
                Tmin_new = TSYM[-1]

            TSYM = np.linspace(Tmax_new, Tmin_new, len(TSYM))
            grid.TSYM = TSYM
            Sr = np.zeros_like(TSYM)
            Hr = np.zeros_like(TSYM)
            Pr = np.zeros_like(TSYM)
            Tb = np.zeros_like(TSYM)
            iterator += 1
            iteration_condition = iterator < settings.maxit

    return PercolationState(
        TSYM=grid.TSYM,
        Sr=Sr,
        Hr=Hr,
        Pr=Pr,
        Tb=Tb,
        explored_tmin=lowest_temperature_reached,
        support_bank=_temperature_grid(grid.TSYM),
    )

def _refine_percolation_temperature(
    state: PercolationState,
    grid: PercolationGrid,
    settings,
    outdict: dict,
    pot,
    phase_symmetric,
    phase_broken,
    CF: float,
    vw: float,
    Tperc_prev: float,
    rtol: float,
    verbose: bool,
) -> tuple[float, PercolationState]:
    """Step 3: iterate with the P-dependent Hubble rate until convergence.
    
    Refine the percolation temperature by solving the
    percolation integral with the Hubble rate and the true vacuum fraction
    calculated in the previous step. The function iteratively
    improves the true vacuum fraction and the Hubble rate until
    convergence is reached."""
    TBROmin = grid.TBROmin
    TBROmax = grid.TBROmax
    TSYM = state.TSYM
    iterator = 1
    iteration_condition = True

    if verbose:
        print("\nPercolation step 3: calculating Tperc with P-dependent Hubble rate")

    while iteration_condition:
        if verbose:
            print(f"Iteration {iterator}, T_perc = {Tperc_prev * CF:2.8g} GeV")

        if state.explored_tmin is None:
            state.explored_tmin = float(TSYM[-1])
        else:
            state.explored_tmin = min(float(state.explored_tmin), float(TSYM[-1]))
        Pprev = state.Pr.copy()
        state.Hr.fill(0)
        state.Pr.fill(0)
        state.Tb.fill(np.nan)
        state.Sr.fill(0)
        is_Pr_converged = False

        for i, T in enumerate(TSYM):
            if not is_Pr_converged:
                eSYM = energyDensity(pot, phase_symmetric, T)
                if i == 0: # High temperature bin
                    # Pprev must be zero here
                    state.Hr[i] = HubbleParameter(eSYM, CF)
                    state.Sr[i] = calcAction(pot, T, phase_symmetric, phase_broken, outdict)
                    try:
                        TBRO = optimize.brentq(
                            Tb_criterion,
                            TBROmin,
                            TBROmax,
                            args=(T, phase_symmetric, phase_broken, pot),
                        )
                        state.Tb[i] = TBRO
                    except ValueError as err:
                        # Sometimes, brentq fails to find a root here. This is usually due to
                        # an inaccuracy in phase tracing, which can be solved by increasing
                        # the diftol parameter in the config file.
                        raise errors.PercolationError(
                            "Error in Tb_criterion: "
                            f"{err}. Please check the input parameters again by hand. "
                            "It might be that the phase tracing is not accurate enough, "
                            "try increasing the diftol parameter in the config file."
                        )
                    eBRO = 0
                else:
                    # When the previous step had negligible broken-phase
                    # fraction, reset to a fresh Tb_criterion value instead
                    # of propagating entropy from a non-existent plasma.
                    _P_ESTABLISHED = 1e-6
                    if Pprev[i - 1] < _P_ESTABLISHED:
                        try:
                            TBRO = optimize.brentq(
                                Tb_criterion, TBROmin, TBROmax,
                                args=(T, phase_symmetric, phase_broken, pot)
                            )
                        except ValueError:
                            TBRO = optimize.brentq(
                                entropy_criterion_SYM_BRO, TBROmin, TBROmax,
                                args=(state.Tb[i - 1], T, TSYM[i - 1], phase_broken, pot)
                            )
                    else:
                        # Use entropy conservation in the symmetric phase between two time steps
                        # to compute the broken phase energy density in step i based on the
                        # broken phase energy density in step i-1.
                        TBRO = optimize.brentq(
                            entropy_criterion_SYM_BRO, TBROmin, TBROmax,
                            args=(state.Tb[i - 1], T, TSYM[i - 1], phase_broken, pot)
                        )
                    eBRO = energyDensity(pot, phase_broken, TBRO)

                    P = Pprev[i]
                    dP = Pprev[i] - Pprev[i - 1]
                    energy_release = eBRO * (P - dP) + dP * eSYM
                    relative_energy_release = energy_release / eBRO
                    if relative_energy_release < 1e-50:
                        # If the energy release between two steps is too small, we don't need to
                        # calculate the broken phase temperature. The following computation in that
                        # case would just try to find an arbitrary argument T in front of a factor zero
                        # being the energy release. It is thus safer to just set TBRO to the value obtained
                        # in the previous step in the entropy conservation.
                        if verbose:
                            print("Energy release is too small to use energy criterion, using T_bro from entropy step")
                        state.Tb[i] = TBRO
                    else:
                        # Use the energy conservation to find the broken phase temperature
                        # based on the energy release between two steps.
                        try:
                            TBRO = optimize.brentq(
                                energy_criterion_BRO,
                                TBROmin,
                                TBROmax,
                                args=(eSYM, eBRO, Pprev[i], Pprev[i] - Pprev[i - 1], phase_broken, pot),
                            )
                            state.Tb[i] = TBRO
                        except ValueError as err:
                            # Sometimes, the energy criterion fails to find a root,
                            # because the energy release is too small.
                            # We try to limit this case by checking the relative
                            # energy release beforehand, but sometimes it still happens.
                            # In that case, we just use the value from the previous step
                            # and print a warning. This is not critical, as the energy
                            # release is small anyway.
                            if verbose:
                                print(
                                    "Warning: energy_criterion failed. This is usually "
                                    "because the relative energy release "
                                    f"{relative_energy_release} is too small. We set it "
                                    "to 0 and continue: ",
                                    err,
                                )
                            state.Tb[i] = state.Tb[i - 1]
                            TBRO = state.Tb[i]
                        except Exception as err:
                            raise errors.PercolationError(
                                "Error in energy_criterion_BRO: "
                                f"{err}. Please check this input parameters again by hand."
                            )

                        eBRO = energyDensity(pot, phase_broken, TBRO)

                    if state.Tb[i] < TSYM[i]:
                        raise ValueError(
                            "There was a problem with the T_bro calculation, "
                            "T_bro < T_sym. This is (usually) not allowed by the "
                            "second law of thermodynamics."
                        )

                state.Sr[i] = calcAction(pot, T, phase_symmetric, phase_broken, outdict)
                state.Hr[i] = HubbleParameter(Pprev[i] * eBRO + (1 - Pprev[i]) * eSYM, CF)
                state.Pr[i] = 1 - np.exp(
                    -percIntegral(
                        TSYM[0 : i + 1],
                        state.Hr[0 : i + 1],
                        state.Sr[0 : i + 1],
                        vw=vw,
                    )
                )
            else:
                state.Sr[i] = np.nan
                rho = energyDensity(pot, phase_symmetric, T)
                state.Hr[i] = HubbleParameter(rho, CF)
                state.Pr[i] = state.Pr[i - 1]

            if verbose:
                print(
                    f"Step {i+1}/{len(TSYM)}, T = {TSYM[i] * CF:2.8g} GeV, "
                    f"P_true = {state.Pr[i]:2.5g}, S_3 = {state.Sr[i] * CF:2.5g} GeV, "
                    f"S_3/T = {state.Sr[i] / T:2.5g}"
                )

            if i >= 1 and state.Pr[i] > settings.f_final and state.Pr[i - 1] > settings.f_final:
                if verbose and not is_Pr_converged:
                    print(
                        "WARNING: P > ",
                        settings.f_final,
                        " at T = ",
                        TSYM[i] * CF,
                        " GeV detected. Stop computation of action from here on "
                        "and only fill in the values for the Hubble rate.",
                    )
                is_Pr_converged = True

        Pint = interpolate.interp1d(TSYM, state.Pr)
        try:
            Tperc = optimize.brentq(lambda T: Pint(T) - settings.f_perc, TSYM[0], TSYM[-1])
        except ValueError as err:
            msg = err.args[0]
            if msg.startswith("f(a) and f(b) must have different signs"):
                if Pint(TSYM[-1]) < settings.f_perc:
                    explored_tmin = float(TSYM[-1]) if state.explored_tmin is None else float(state.explored_tmin)
                    current_tmin = float(TSYM[-1])
                    collapse_tol = 1e-12 * max(abs(explored_tmin), abs(current_tmin), 1.0)
                    if explored_tmin + collapse_tol < current_tmin:
                        raise errors.PercolationError(
                            "The percolation-temperature refinement became numerically unstable: "
                            "the support grid previously explored lower temperatures down to "
                            f"{explored_tmin * CF:2.8g} GeV, but the final step-3 support "
                            "grid shrank back to Tmin = "
                            f"{current_tmin * CF:2.8g} GeV before failing with "
                            f"P(Tmin) = {Pint(TSYM[-1]):2.5g} < {settings.f_perc:2.5g}. "
                            "This usually indicates a spurious 0->1 jump in P(T); rerun "
                            "with higher percolation resolution."
                        )
                    raise errors.TooMuchSupercoolingError(
                        "The percolation temperature could not be found because the "
                        "true vacuum fraction only reaches "
                        f"{Pint(TSYM[-1])} < {settings.f_perc} at Tmin = "
                        f"{TSYM[-1] * CF} GeV."
                    )
            raise errors.PercolationError(err)

        except Exception as err:
            msg = (
                "Warning: could not compute Tperc after the third approximation step "
                "with variable P in Hubble rate: "
                f"{err}. Set the percolation temperature to the one from the previous step."
            )
            if verbose:
                print(msg)
            Tperc = Tperc_prev

        if np.abs(Tperc - Tperc_prev) / Tperc < rtol:
            if state.Pr[0] <= settings.f_start and state.Pr[-1] >= settings.f_final:
                jump_indices = np.where((state.Pr[:-1] == 0) & (state.Pr[1:] == 1))[0]
                has_jump_0_to_1 = jump_indices.size > 0
                boundary_n = min(settings.max_boundary_n, len(state.Pr) - 1)
                if has_jump_0_to_1:
                    max_index = jump_indices[0]
                    min_index = jump_indices[0] + 1
                    tsym_jump_max = TSYM[max_index]
                    tsym_jump_min = TSYM[min_index]
                    if verbose:
                        print(
                            "Warning: Pr includes a 0 followed by 1, i.e. a "
                            "discontinuous jump in true vacuum fraction."
                        )
                        print(
                            f"Jump occurs between the indices {max_index} and {min_index}, "
                            f"TSYM values: {tsym_jump_min} and {tsym_jump_max}. "
                            "Set these as new boundaries."
                        )
                    if tsym_jump_max / tsym_jump_min - 1 > 1e-10:
                        TSYM = np.linspace(tsym_jump_max, tsym_jump_min, len(TSYM))
                        state.TSYM = TSYM
                        state.Sr = np.zeros_like(TSYM)
                        state.Hr = np.zeros_like(TSYM)
                        state.Pr = np.zeros_like(TSYM)
                        state.Tb = np.zeros_like(TSYM)
                        Tperc_prev = Tperc
                        iterator += 1
                        iteration_condition = iterator < settings.maxit
                    else:
                        if verbose:
                            print(
                                "WARNING: Even though there is a jump in P_true, the "
                                "temperature range is already very small. We will not "
                                "adjust it further and call the percolation temperature "
                                "calculation converged."
                            )
                            print(
                                "\nConverged after "
                                f"{iterator} iterations to Tperc = "
                                f"{Tperc * CF:2.8g} GeV"
                            )
                        iteration_condition = False
                elif state.Pr[boundary_n] == 0 or state.Pr[-boundary_n] == 1:
                    if verbose and iterator != settings.maxit:
                        print(
                            "Warning: the temperature range for the percolation "
                            "temperature computation was chosen too wide in iteration ",
                            iterator,
                            ". Decrease it!",
                        )
                    if state.Pr[boundary_n] == 0:
                        Tmax_new = TSYM[np.where(state.Pr < settings.f_start)[0][-1]]
                    else:
                        Tmax_new = TSYM[0]

                    if state.Pr[-boundary_n] == 1:
                        Tmin_new = TSYM[np.where(state.Pr > settings.f_final)[0][0]]
                    else:
                        Tmin_new = TSYM[-1]

                    if Tmax_new < Tmin_new:
                        raise errors.PercolationError(
                            "The temperature range for the percolation temperature "
                            "computation was messed up. It is not clear how this could "
                            "have happened."
                        )
                    TSYM = np.linspace(Tmax_new, Tmin_new, len(TSYM))
                    state.TSYM = TSYM
                    state.Sr = np.zeros_like(TSYM)
                    state.Hr = np.zeros_like(TSYM)
                    state.Pr = np.zeros_like(TSYM)
                    state.Tb = np.zeros_like(TSYM)
                    Tperc_prev = Tperc
                    iterator += 1
                    iteration_condition = iterator < settings.maxit
                else:
                    if verbose:
                        print(
                            "\nConverged after "
                            f"{iterator} iterations to Tperc = "
                            f"{Tperc * CF:2.8g} GeV"
                        )
                    iteration_condition = False
            else:
                # The temperature range is not large enough to find a percolation temperature, we need to increase it
                if verbose:
                    print(
                        "Warning: even though a percolation temperature was previously "
                        f"found (Tperc = {Tperc * CF:2.8g} GeV), it is not clear "
                        "whether the phase transition can finalize. The temperature "
                        "range will therefore be increased."
                    )
                    print("This was iteration", iterator, "of the third step of the percolation calculation.")
                    if iterator == settings.maxit - 1:
                        print(
                            "This was the penultimate iteration, the temperature range "
                            "will be set to the maximum possible range."
                        )
                    elif iterator == settings.maxit:
                        print("This was the last iteration. We stop here and return an error.")
                    else:
                        print("We will increase the temperature range and try again.")

                if state.Pr[0] > settings.f_start:
                    Tmax_new = TSYM[0] + (grid.tmax - TSYM[0]) * settings.rel_increment
                else:
                    Tmax_new = TSYM[0]

                if state.Pr[-1] < settings.f_final:
                    if iterator == settings.maxit - 1:
                        # If we are at the last iteration, we set the lower temperature boundary
                        # all the way down to the minimal possible temperature allowed by the
                        # potential. This is to ensure that we do not miss the temperature at which
                        # the transition completes.
                        Tmin_new = grid.tmin
                    elif iterator == settings.maxit:
                        last_evaluated_tmin = float(TSYM[-1])
                        lower_bound_reached = (
                            abs(last_evaluated_tmin - float(grid.tmin))
                            <= 1e-12 * max(abs(last_evaluated_tmin), abs(float(grid.tmin)), 1.0)
                        )
                        if lower_bound_reached:
                            raise errors.EternalInflationError(
                                "The transition cannot completely finish because the true "
                                "vacuum fraction at the lowest evaluated temperature Tmin = "
                                f"{last_evaluated_tmin * CF:2.5g} GeV computed in step 3 "
                                "of the percolation computation stays below f_final: "
                                f"P(Tmin) = {state.Pr[-1]:2.5g} < {settings.f_final:2.5g}."
                            )
                        raise errors.PercolationError(
                            "The percolation computation reached the iteration limit before "
                            "the low-temperature tail was resolved: the lowest evaluated "
                            f"support point was Tmin = {last_evaluated_tmin * CF:2.5g} GeV "
                            f"with P(Tmin) = {state.Pr[-1]:2.5g}, while the configured lower "
                            f"bound is {grid.tmin * CF:2.5g} GeV. Increase the percolation "
                            "resolution to decide whether the transition completes or enters "
                            "eternal inflation."
                        )
                    else:
                        Tmin_new = TSYM[-1] - (TSYM[-1] - grid.tmin) * settings.rel_increment
                else:
                    Tmin_new = TSYM[-1]

                TSYM = np.linspace(Tmax_new, Tmin_new, len(TSYM))
                state.TSYM = TSYM
                state.Sr = np.zeros_like(TSYM)
                state.Hr = np.zeros_like(TSYM)
                state.Pr = np.zeros_like(TSYM)
                state.Tb = np.zeros_like(TSYM)
                Tperc_prev = Tperc
                iterator += 1
                iteration_condition = iterator < settings.maxit
        else:
            if verbose:
                print(
                    "Percolation temperature T_perc = ",
                    Tperc,
                    "was found, but it is not close enough to the previous one (",
                    Tperc_prev,
                    ") at iteration ",
                    iterator,
                    ". Iterate one more time with the P(t) from the previous step and see if T_perc converges.",
                )
            Tperc_prev = Tperc
            iterator += 1
            iteration_condition = iterator < settings.maxit

    state.support_bank = _temperature_grid(state.TSYM)
    return Tperc, state
