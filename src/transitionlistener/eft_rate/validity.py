"""Run-time validity monitor for the dimensionally reduced (3D EFT) rate mode.

The three-dimensional effective theory is built by integrating out the modes
that are heavy at a given temperature. Whether that was legitimate is not
knowable in advance -- it depends on where in field space the bounce actually
went, which is an output. This module collects the ratios that decide it and
turns them into a verdict, so that a result is labelled trusted only when the
modes we integrated out really were heavy in the region that mattered.

A model (or the generated matching module beside it) supplies the numbers;
this module holds the thresholds and the bookkeeping.

Two choices shape the checks:

* The hard-scale test asks whether the reduction is *defined*: each species
  against the convergence radius of its own high-temperature expansion, the
  lowest non-zero Matsubara mode -- 2 pi T for bosons, pi T for fermions -- with
  a single pass/fail at one. Whether it is *accurate* is a separate question,
  answered by the next point.
* Accuracy of the dimension-four truncation is measured, not inferred: the
  first dropped operator, (c6/8) phi3^6, integrated over the bounce. Its size
  against the leading-order action is the `dim6` check, and the shift itself is
  quoted as an uncertainty (`ExponentUncertainty`).

The next-to-leading-order correction carries an error bar for the order after
it, |Delta S| times the 3D loop-expansion parameter g3^2/(pi m3), rather than a
cut; the 25% size flag is a marker only (`nlo_is_large`).

Which checks bear on a result depends on how it was computed, so `evaluate`
reads its thresholds from `thresholds_for(branch, order)`:

* ``soft_separation`` asks whether A_0 may be integrated out. It counts only for
  a result in the ultrasoft theory; in the soft theory, which keeps A_0 and is
  what the production rate uses, it is recorded and never warns.
* ``expansion_3d`` sets the next-order error bar of an NLO result, so there it
  is pass/fail at one, where the error bar equals the correction. A
  leading-order result has no error bar, and keeps the warning band.

Documentation: https://tasillo.de/TransitionListener/

Authors:
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

Verdict = Literal["ok", "warn", "fail"]

#: check name -> (warn above, fail above, one-line meaning): the strictest
#: reading, an ultrasoft leading-order result. `thresholds_for` relaxes it.
DEFAULT_THRESHOLDS: dict[str, tuple[float, float, str]] = {
    # is the high-temperature expansion defined where the bounce goes? Each
    # species against its own lowest non-zero Matsubara mode; pass/fail only,
    # accuracy is the dim6 check
    "hard": (1.0, 1.0, "heaviest kept boson over its hard scale 2*pi*T"),
    "hard_fermion": (1.0, 1.0, "heaviest fermion over its hard scale pi*T"),
    # is the soft (Debye) scale actually separated from the bosonic hard scale?
    "debye": (0.5, 1.0, "Debye mass over the bosonic hard scale 2*pi*T"),
    # is A_0 heavy enough to be integrated out relative to the light 3D modes?
    "soft_separation": (0.5, 1.0, "lightest 3D mass over the Debye mass"),
    # loop expansion parameter of the 3D theory itself
    "expansion_3d": (0.3, 1.0, "3D loop expansion parameter g3^2/(pi m_light)"),
    # the first operator the reduction drops, measured on the bounce
    "dim6": (0.05, 0.25, "dimension-six action shift over the LO action, |dS6|/S3"),
}

BRANCHES = ("soft", "ultrasoft")
ORDERS = ("LO", "NLO")


def thresholds_for(branch: str = "soft", order: str = "LO") -> dict[str, tuple[float, float, str]]:
    """The thresholds that apply to a result from ``branch`` at ``order``.

    ``branch`` is the three-dimensional theory the rate was computed in: "soft"
    keeps A_0, "ultrasoft" has integrated it out. ``order`` is "LO" or "NLO"
    (next-to-leading order with its error bar).
    """
    if branch not in BRANCHES:
        raise ValueError(f"branch must be one of {BRANCHES}, not {branch!r}")
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}, not {order!r}")
    table = dict(DEFAULT_THRESHOLDS)
    if branch == "soft":
        table["soft_separation"] = (
            math.inf, math.inf,
            "lightest 3D mass over the Debye mass; recorded only, since the soft "
            "theory keeps A_0")
    if order == "NLO":
        table["expansion_3d"] = (
            1.0, 1.0,
            "3D loop expansion parameter g3^2/(pi m_light); its size is the NLO "
            "error bar, which equals the correction at one")
    return table


#: checks whose ingredients are not available yet are simply absent from a
#: report rather than silently passing; this records why.
_MISSING_IS_NOT_PASSING = (
    "A check that could not be evaluated is reported as missing, never as "
    "passing. Read `missing` before trusting a verdict."
)


@dataclass
class ValidityReport:
    """The outcome of one validity evaluation."""

    ratios: dict[str, float] = field(default_factory=dict)
    verdict: Verdict = "ok"
    messages: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    #: context, e.g. {"T": ..., "phi_over_T": ...}, carried through to output
    context: dict[str, float] = field(default_factory=dict)
    #: the thresholds the verdict was reached with
    thresholds: dict[str, tuple[float, float, str]] = field(
        default_factory=lambda: dict(DEFAULT_THRESHOLDS))

    @property
    def trusted(self) -> bool:
        """True only if every evaluated check passed and none is missing."""
        return self.verdict == "ok" and not self.missing

    def worst(self) -> tuple[str, float] | None:
        """The check furthest into its warning band, as (name, ratio)."""
        if not self.ratios:
            return None
        scored = []
        for name, value in self.ratios.items():
            warn, _fail, _ = self.thresholds.get(name, (1.0, 1.0, ""))
            if warn > 0:
                scored.append((value / warn, name, value))
        if not scored:
            return None
        _, name, value = max(scored)
        return name, value

    def as_dict(self) -> dict[str, float | str]:
        """Flat mapping suitable for writing into a scan output row."""
        out: dict[str, float | str] = {
            f"eft_{name}": value for name, value in self.ratios.items()
        }
        out["eft_verdict"] = self.verdict
        for key, value in self.context.items():
            out[f"eft_{key}"] = value
        return out


def evaluate(
    ratios: dict[str, float | None],
    thresholds: dict[str, tuple[float, float, str]] | None = None,
    context: dict[str, float] | None = None,
    *,
    branch: str = "soft",
    order: str = "LO",
) -> ValidityReport:
    """Turn a set of measured ratios into a verdict.

    Parameters
    ----------
    ratios
        Mapping from check name (see ``DEFAULT_THRESHOLDS``) to the measured
        value. A value of ``None`` means the check could not be evaluated; it
        is recorded as missing and never counted as a pass.
    thresholds
        Optional explicit table; overrides ``branch`` and ``order``.
    context
        Optional extra numbers (temperature, phi/T, ...) carried into the report.
    branch, order
        How the result was computed, see `thresholds_for`. The default is the
        soft theory at leading order, the stricter of the two orders.
    """
    table = thresholds_for(branch, order) if thresholds is None else thresholds
    report = ValidityReport(context=dict(context or {}), thresholds=dict(table))
    verdict: Verdict = "ok"

    for name, value in ratios.items():
        if name not in table:
            raise KeyError(
                f"unknown validity check {name!r}; known checks: {sorted(table)}"
            )
        if value is None:
            report.missing.append(name)
            continue
        value = float(value)
        report.ratios[name] = value
        warn, fail, meaning = table[name]
        if value > fail:
            verdict = "fail"
            report.messages.append(
                f"{name} = {value:.3g} exceeds {fail:g} ({meaning})"
            )
        elif value > warn and verdict != "fail":
            verdict = "warn"
            report.messages.append(
                f"{name} = {value:.3g} exceeds {warn:g} ({meaning})"
            )

    report.verdict = verdict
    if report.missing:
        report.messages.append(
            "not evaluated: " + ", ".join(sorted(report.missing))
            + ". " + _MISSING_IS_NOT_PASSING
        )
    return report


#: Where `soft_separation` alone cannot decide. Below the first number the
#: light 3D modes are well under the Debye mass and A_0 may be integrated out;
#: above the second it clearly may not. Between them the ratio abstains and the
#: decision has to come from the stability test instead. The numbers are the
#: warn/fail thresholds of the `soft_separation` check, kept in step with it
#: deliberately: the branch choice and the validity warning are the same
#: statement asked twice.
ULTRASOFT_SEPARATION_OK = DEFAULT_THRESHOLDS["soft_separation"][0]
ULTRASOFT_SEPARATION_BAD = DEFAULT_THRESHOLDS["soft_separation"][1]


def dim6_ratio(delta_S6: float, S3_over_T: float) -> float:
    """``|Delta S_6| / S_3``: the dropped dimension-six operator against the action.

    ``delta_S6`` is ``4 pi int r^2 (c6/8) phi3^6`` on the leading-order bounce
    (:func:`transitionlistener.eft_rate.abelian_higgs_3d.dim6_action_shift`). The thresholds are a
    stated choice, 5% (warn) and 25% (fail) of the leading-order exponent: the
    fail level is the size at which the next-to-leading-order correction is
    flagged, so a dropped operator that large is treated the same way.
    """
    if S3_over_T <= 0.0:
        return float("inf")
    return abs(delta_S6) / S3_over_T


#: |Delta S| / S_3 above which a next-to-leading-order result is marked. A
#: marker, not a cut: the error bar below is what is quoted.
NLO_LARGE_FRACTION = 0.25


def nlo_is_large(delta_S: float, S3_over_T: float) -> bool:
    """True where the NLO correction exceeds 25% of the leading-order action."""
    return abs(delta_S) > NLO_LARGE_FRACTION * S3_over_T


@dataclass
class ExponentUncertainty:
    """Uncertainties on a rate exponent, in exponent units, and on T_perc.

    ``nnlo``
        The order after the computed next-to-leading one, estimated as
        ``|Delta S| g3^2/(pi m3)``: the correction times the loop-expansion
        parameter of the three-dimensional theory it was computed in (the
        `expansion_3d` ratio). Zero at leading order, where there is no
        correction to scale.
    ``dim6``
        ``|Delta S_6|``, the measured effect of the first dropped operator.

    The renormalisation-scale band is not here: it comes from rerunning at
    other scales, not from one temperature.
    """

    nnlo: float = 0.0
    dim6: float = 0.0

    @classmethod
    def from_ratios(cls, delta_S: float, expansion_3d: float,
                    delta_S6: float) -> "ExponentUncertainty":
        return cls(nnlo=abs(delta_S) * expansion_3d, dim6=abs(delta_S6))

    def on_tperc(self, beta_over_H: float) -> dict[str, float]:
        """Fractional uncertainties on T_perc, delta T/T = delta S / (beta/H).

        ``beta_over_H`` must be ``T dS/dT`` at T_perc (TransitionListener's
        ``betaH_S3``), the slope that converts a shift of the exponent at fixed
        T into a shift of the temperature where it reaches a given value.
        """
        if not beta_over_H > 0.0:
            raise ValueError("beta/H must be positive to propagate to T_perc")
        return {"nnlo": self.nnlo / beta_over_H, "dim6": self.dim6 / beta_over_H}
