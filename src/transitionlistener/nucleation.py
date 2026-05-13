"""Nucleation-temperature search and tunneling-action bookkeeping.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

import numpy as np
from scipy import optimize

from transitionlistener import errors
from transitionlistener.pathDeformation import bounceAction


def findTunnelingTmax(DV: callable, Tmin: float, Tmax: float, tol=1e-15, maxiter=100):
    """Largest temperature where tunneling is energetically allowed."""

    if DV(Tmax) > 0:
        return Tmax
    try:
        return optimize.brentq(DV, Tmin, Tmax, xtol=tol, maxiter=maxiter)
    except ValueError:
        try:
            return optimize.brentq(DV, 0.5 * (Tmin + Tmax), Tmax, xtol=tol, maxiter=maxiter)
        except ValueError:
            return Tmin


def findTminFromUShapedNucl(
    Tmin,
    Tmax,
    args,
    start_phase,
    end_phase,
    outdict,
    Ttol,
    maxiter,
    CF,
    verbose,
    nuclCriterion,
):
    """Find the action minimum when the nucleation criterion is U-shaped."""

    def maximizer(x, *criterion_args):
        return -evalNucleationCriterion(x, *criterion_args)

    try:
        Ttest = np.nan
        for temperature in np.linspace(Tmin, Tmax, 100, endpoint=False)[1:]:
            criterion = evalNucleationCriterion(temperature, *args)
            if not np.isinf(criterion):
                Ttest = temperature
                break

        if np.isnan(Ttest):
            if any(outdict[T]["trantype"] == 2 for T in outdict):
                max_key = max(T for T in outdict if outdict[T]["trantype"] == 2)
                if verbose:
                    print(
                        "Found a second-order transition at T = ",
                        max_key * CF,
                        "GeV while scanning a weak first-order candidate.",
                    )
                outdict[max_key]["Tcrit"] = outdict[max_key]["Tnuc"] = max_key
                return None
            if all(
                evalNucleationCriterion(T, *args) < 0
                if Tmin <= T <= Tmax
                else True
                for T in outdict
            ):
                raise errors.NucleationError("Nucleation criterion failed: too low at all temperatures.")
            raise errors.NucleationError(
                "Nucleation criterion failed: no finite criterion found between Tmin and Tmax."
            )

        result = optimize.minimize_scalar(
            maximizer,
            bracket=(Tmin, Ttest, Tmax),
            args=args,
            method="brent",
            options={"xtol": (Tmax - Tmin) * 1e-5, "maxiter": maxiter, "disp": 0},
        )
        if not result.success:
            raise errors.NucleationError("Nucleation criterion minimization failed: " + str(result.message))

        Tmin_u = float(np.squeeze(result.x))
        action_u = outdict[Tmin_u]["action"]
        ncMin_u = nuclCriterion(action_u, Tmin_u, start_phase, end_phase)
        if verbose:
            print("Finding Tmin_u done! Tmin_u = ", Tmin_u * CF, "GeV")
            print("action_u = ", action_u * CF, "GeV")
            print("ncMin_u = ", ncMin_u)
        if ncMin_u < 0:
            raise errors.NucleationError(
                "Nucleation criterion failed because the nucleation rate is too low at all temperatures."
            )
        return Tmin_u
    except Exception as err:
        raise errors.NucleationError("Nucleation calculation failed: " + str(err)) from err


def computeNucleationTemperature(
    V,
    dV,
    d2V,
    start_phase,
    end_phase,
    CF,
    Ttol=1e-8,
    maxiter=100,
    phitol=1e-8,
    overlapAngle=45.0,
    nuclCriterion=lambda S, T, *_: 140 - S / (T + 1e-100),
    approximate_strength_threshold=1e-4,
    verbose=False,
    fullTunneling_params=None,
    allow_second_order_fallback=True,
):
    """Find the instanton and nucleation temperature for one transition."""

    outdict = {}
    fullTunneling_params = {} if fullTunneling_params is None else fullTunneling_params
    args = (
        start_phase,
        end_phase,
        V,
        dV,
        d2V,
        phitol,
        overlapAngle,
        nuclCriterion,
        fullTunneling_params,
        verbose,
        outdict,
        CF,
    )
    Tmin = max(start_phase.Tmin, end_phase.Tmin)

    def DV(T):
        from transitionlistener.phases import findLocalMinimum

        x0 = findLocalMinimum(start_phase.valAt(T), T, dV, d2V)
        x1 = findLocalMinimum(end_phase.valAt(T), T, dV, d2V)
        if np.sqrt(np.sum((x0 - start_phase.valAt(T)) ** 2)) > 1:
            x0 = start_phase.valAt(T)
        if np.sqrt(np.sum((x1 - end_phase.valAt(T)) ** 2)) > 1:
            x1 = end_phase.valAt(T)
        dist = np.sqrt(np.dot(x0 - x1, x0 - x1))
        return V(x0, T) - (V(x1, T) + dist * 1e-3)

    Tmax = findTunnelingTmax(DV, Tmin, end_phase.Tmax)
    if not checkNucleationPossibility(
        Tmin,
        Tmax,
        args,
        outdict,
        V,
        dV,
        start_phase,
        end_phase,
        CF,
        approximate_strength_threshold,
        verbose,
    ):
        return dict(
            Tcrit=Tmax,
            Tnuc=Tmax,
            Tperc=Tmax,
            low_vev=end_phase.valAt(Tmax),
            high_vev=start_phase.valAt(Tmax),
            low_phase=end_phase.key,
            high_phase=start_phase.key,
            action=np.nan,
            full_tunneling_info=np.nan,
            instanton=np.nan,
            trantype=2,
        )

    ncrit_Tmin = evalNucleationCriterion(Tmin, *args)
    ncrit_Tmax = evalNucleationCriterion(Tmax, *args)
    if ncrit_Tmax >= 0:
        if verbose:
            print(f"Nucl. crit fulfilled at Tmax = {Tmax * CF}!")
        rdict = outdict[Tmax]
        rdict["Tcrit"] = Tmax
        if rdict["trantype"] != 2:
            rdict["full_tunneling_info"] = _filtered_tunneling_history(outdict)
        return rdict

    if ncrit_Tmin < 0:
        if verbose:
            print("Nucleation criterion not fulfilled at Tmin")
            print("The bounce action might be U-shaped. Try to find the minimum of it")
            print("between Tmin = ", Tmin * CF, "GeV and Tmax = ", Tmax * CF, "GeV.")
        Tmin = findTminFromUShapedNucl(
            Tmin,
            Tmax,
            args,
            start_phase,
            end_phase,
            outdict,
            Ttol,
            maxiter,
            CF,
            verbose,
            nuclCriterion,
        )
        if Tmin is None:
            return dict(
                Tcrit=Tmax,
                Tnuc=Tmax,
                Tperc=Tmax,
                low_vev=end_phase.valAt(Tmax),
                high_vev=start_phase.valAt(Tmax),
                low_phase=end_phase.key,
                high_phase=start_phase.key,
                action=np.nan,
                full_tunneling_info=np.nan,
                instanton=np.nan,
                trantype=2,
            )

    try:
        Tnuc = optimize.brentq(evalNucleationCriterion, Tmin, Tmax, args=args, xtol=Ttol, maxiter=maxiter, disp=False)
        if verbose:
            print("Brentq done after first try. Tnuc = ", Tnuc * CF, "GeV")
    except ValueError:
        print("Brentq failed for first try, retrying with larger Ttol.")
        try:
            Tnuc = optimize.brentq(
                evalNucleationCriterion,
                Tmin,
                Tmax,
                args=args,
                xtol=Ttol * 10,
                maxiter=maxiter,
                disp=False,
            )
            if verbose:
                print("Brentq done after second try! Tnuc = ", Tnuc * CF, "GeV")
        except Exception as err:
            if verbose:
                print("Brentq failed again after increasing the numerical tolerance by a factor 10.")
            raise errors.NucleationError("Nucleation criterion failed due to unknown reason.") from err

    if Tnuc not in outdict:
        raise errors.NucleationError(
            "Tnuc not in outdict. This can only happen if the nucleation criterion "
            "could not be fulfilled at any temperature."
        )
    rdict = outdict[Tnuc]
    rdict["full_tunneling_info"] = _filtered_tunneling_history(outdict)

    if "trantype" in rdict and rdict["trantype"] > 0:
        if rdict["trantype"] == 2:
            if verbose:
                print("The transition is actually second-order because the action is zero at the transition.")
            rdict["Tcrit"] = rdict["Tnuc"] = Tnuc
            return rdict
        rdict["Tcrit"] = Tmax
        return rdict

    if "trantype" in rdict and rdict["trantype"] == 0:
        keys = sorted(key for key in outdict if key != "full_tunneling_info" and key < Tnuc)
        max_key = keys[-1] if keys else None
        if max_key is not None and outdict[max_key]["trantype"] == 2:
            if not allow_second_order_fallback:
                raise errors.NucleationError(
                    "Nucleation criterion converged to a pathological trantype = 0 candidate, "
                    "and the only nearby fallback is a cached second-order entry just below Tnuc."
                )
            if verbose:
                print(
                    "The found transition had an infinite action. But found a second-order "
                    "transition just below Tnuc. Returning it. Tcrit = ",
                    max_key * CF,
                    "GeV",
                )
            rdict = outdict[max_key]
            rdict["Tcrit"] = rdict["Tnuc"] = max_key
            return rdict
        if verbose:
            print(
                "The transition has action = 0 and trantype = 0, but no second-order "
                "transition was found just below Tnuc. This is likely numerical."
            )
        return rdict

    raise errors.NucleationError(
        "The nucleation computation failed: either the barrier is too high or the transition is second order."
    )


def checkNucleationPossibility(
    Tmin,
    Tmax,
    args,
    outdict,
    V,
    dV,
    start_phase,
    end_phase,
    conversionFactor,
    apprx_strength_thr,
    verbose,
) -> bool:
    """Quickly reject hidden second-order or negligibly weak transitions."""

    nuclCritTmax = evalNucleationCriterion(Tmax, *args)
    if np.isinf(nuclCritTmax) and Tmax in outdict:
        rdict = outdict[Tmax]
        if "trantype" in rdict and rdict["trantype"] == 2:
            if verbose:
                print("The transition is actually second-order because the action is zero at the transition.")
            return False

    def approx_strength(T):
        DV = V(start_phase.valAt(T), T) - V(end_phase.valAt(T), T)
        return np.abs(DV) / T**4

    approx_strength_range = np.array([approx_strength(T) for T in np.linspace(Tmin, Tmax, 100)])
    if np.all(approx_strength_range < apprx_strength_thr):
        if verbose:
            print(f"PT strength {np.max(approx_strength_range)} below threshold: {apprx_strength_thr}.")
            print(f"This is treated as a second-order transition at Tmax = {Tmax * conversionFactor} GeV.")
        return False
    return True


def evalNucleationCriterion(
    T,
    start_phase,
    end_phase,
    V,
    dV,
    d2V,
    phitol,
    overlapAngle,
    nuclCriterion,
    fullTunneling_params,
    verbose,
    outdict,
    conversionFactor,
):
    """Evaluate the nucleation criterion and cache the action in ``outdict``."""

    try:
        T = T[0]
    except Exception:
        pass

    if T in outdict:
        return nuclCriterion(outdict[T]["action"], T, start_phase, end_phase)

    from transitionlistener.phases import findLocalMinimum

    x0 = findLocalMinimum(start_phase.valAt(T), T, dV, d2V)
    if np.sqrt(np.sum((x0 - start_phase.valAt(T)) ** 2)) > 1:
        x0 = start_phase.valAt(T)
    V0 = V(x0, T)

    if end_phase.Tmin > T:
        if verbose:
            print(
                "Phase %s not valid at T = %g" % (end_phase.key, T * conversionFactor),
                "GeV < Tmin = ",
                end_phase.Tmin * conversionFactor,
                "GeV",
            )
        return np.inf
    if end_phase.Tmax < T:
        if verbose:
            print(
                "Phase %s not valid at T = %g" % (end_phase.key, T * conversionFactor),
                "GeV > Tmax = ",
                end_phase.Tmax * conversionFactor,
                "GeV",
            )
        return -np.inf

    x1 = findLocalMinimum(end_phase.valAt(T), T, dV, d2V)
    if np.sqrt(np.sum((x1 - end_phase.valAt(T)) ** 2)) > 1:
        x1 = end_phase.valAt(T)
    if V(x1, T) >= V0 and verbose:
        print(
            "Tunneling energetically not possible. Tmax might be too high or the "
            "numerical accuracy is not sufficient. This is not a problem, just a warning."
        )

    tdict = dict(low_vev=x1, high_vev=x0, Tnuc=T, low_phase=end_phase.key, high_phase=start_phase.key)
    bounceAction(
        T,
        V,
        dV,
        outdict,
        tdict,
        verbose=verbose,
        conversionFactor=conversionFactor,
        **fullTunneling_params,
    )
    return nuclCriterion(outdict[T]["action"], T, start_phase, end_phase)


def _filtered_tunneling_history(outdict: dict) -> dict:
    """Return cached tunneling data without large instanton payloads."""

    filtered = {T: {key: value for key, value in payload.items() if key != "instanton"} for T, payload in outdict.items()}
    return dict(sorted(filtered.items()))
