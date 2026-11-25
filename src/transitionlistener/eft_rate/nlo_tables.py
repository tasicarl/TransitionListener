r"""The NLO correction :math:`\Delta S` to the 3D action: computing it per temperature, tabulating it on a grid,
and interpolating the grid at run time.

The physics is that of :mod:`transitionlistener.eft_rate.abelian_higgs_3d` (eqs. A.1-A.4 of the paper). Three layers, each usable alone.

**One temperature** (:func:`fast_row`, :func:`mixing_row`, :func:`delta_S_of_row`). The model supplies the matched
parameters (``soft_parameters(T)``) and :math:`V_3^{\rm NLO}`, :math:`Z_3^{\rm NLO}` (``two_loop_potential(p)``,
``two_loop_kinetic(p)``). :func:`fast_row` solves the bounce of :math:`V_3` and evaluates every term of :math:`E`,
eq. (A.2), but :math:`\tfrac12\Delta_{\rm mix}` and :math:`S_6` (``abelian_higgs_3d.nlo_assembly``); :func:`mixing_row` computes
:math:`\Delta_{\rm mix}` (``abelian_higgs_3d.mixing_log_det``), the expensive part; :func:`delta_S_of_row` combines them into
:math:`\Delta S`, eq. (A.4) (``abelian_higgs_3d.action_seam_correction``). The two parts of :math:`S_6`, ``dS6`` and ``dSkin``
(``abelian_higgs_3d.dim6_action_shift``, ``abelian_higgs_3d.dim6_kinetic_action_shift``), are stored beside each row and are not part of
:math:`\Delta S`. The fluctuation problem is solved with T as the unit of energy, since the radial solvers have
absolute tolerances.

**One table** (:func:`build_table`, :func:`build_node`). The rows of one coupling point at a list of temperatures,
in a JSON file; resumable, every finished temperature is written at once. :func:`build_node` chooses the temperatures
where :math:`S_{\rm LO} = S_3/T` takes given values (:func:`temperature_at_action`), for each factor of the matching
scale :math:`\mu_4`.

**The grid** (:func:`assemble_grid`, :class:`NLOGrid`). Node tables collected into one npz file holding

.. math:: r = \Delta S / S_{\rm LO}\quad\text{at fixed}\quad S_{\rm LO} = S_3/T

on a common list of :math:`S_{\rm LO}` for every node. At fixed couplings :math:`S_{\rm LO}` is a monotonic function
of :math:`T/\mu_0` alone (the model has one dimensionful input :math:`\mu_0`), so one table holds for every
:math:`v`. The nodes in the couplings are a list of points: one coupling (g) is interpolated by a cubic spline, two
(g, y) by Clough-Tocher on a Delaunay triangulation. A node whose bounce search stops short at the cold end records
that floor, and a query below the floor of a node it is interpolated from counts as outside. The file records the
conventions of the matching, and :func:`correction_for_model` refuses a run whose configuration differs.

**At run time** (:func:`correction_for_model`, :class:`GridCorrection`): the object a model holds in
``nlo_correction``, :math:`\Delta S` as a function of :math:`S_{\rm LO}` at the model's couplings. Inside the range of
:math:`S_{\rm LO}` it interpolates. Outside the couplings of the grid it raises :class:`NLOGridError`. Outside the
range of :math:`S_{\rm LO}` it raises too, unless the rate does not matter there:

* above ``S_max``, if at the grid's hot end the rate is already below :math:`H^4` by a factor ``e^RATE_MARGIN``;
* below ``S_min``, if at the grid's cold end the rate is already above :math:`H^4` by that factor.

Only there is the ratio continued, bounded and with a continuous slope (:func:`_continued`). With
``nucleationConf.include_dim6`` :math:`S_6` enters the exponent :math:`E` of eq. (A.2) (:func:`_with_dim6`).

Documentation: https://tasillo.de/TransitionListener/

Authors:
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import time
import warnings

import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator, CubicSpline, PchipInterpolator, make_interp_spline
from scipy.spatial import Delaunay, QhullError

from transitionlistener import errors

from transitionlistener.eft_rate import abelian_higgs_3d
from transitionlistener.helper_functions import IntegrationError

#: Version of the assembly of E stored in every table; a table of another version is refused. This version: radial
#: channel in units of T, no kinetic term, the zero-mode Jacobian handled in Delta S (eq. A.4),
#: ``A_dyn = sqrt|lambda_-|/(2 pi)``, and the J = 0 sector of abelian_higgs_3d.j0_sector.
ASSEMBLY = 5

#: Factor e^RATE_MARGIN by which the rate has to lie below (above) H^4 at the hot (cold) end of a grid before the
#: temperatures beyond it count as not mattering. Percolation needs the rate within a few e-folds of H^4.
RATE_MARGIN = 20.0

#: The conventions a grid fixes and a run has to share: file header key -> nucleationConf attribute. g3^2 is matched
#: at matching_order.
CONVENTIONS = {"mu3_choice": "mu3_choice", "input_scheme": "input_scheme", "matching_order": "matching_order",
               "temporal_coupling": "temporal_coupling", "g3sq_order": "matching_order"}
#: The value of a convention in a table that does not record it.
CONVENTION_UNRECORDED = {"temporal_coupling": "2g3sq", "g3sq_order": "LO"}

#: What solving the leading-order bounce at a temperature can raise: no barrier, no running to that scale, or a shooting
#: that fails. These temperatures have no action; they are not errors of a run.
NO_BOUNCE = (ValueError, RuntimeError, ArithmeticError, errors.TunnelingError, IntegrationError)

#: The reduced Planck mass in GeV.
M_PL_GEV = 2.435e18

GRID_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tab_data", "nlo_grids")


class NLOGridError(ValueError):
    """A run asked the NLO grid for something the grid does not hold, or its conventions differ from the grid's."""


# ----------------------------------------------------------------------
# one table
# ----------------------------------------------------------------------
def load_model(model, potential="specific_potential"):
    """The potential class of a model with a 3D rate: ``model`` is the path of a model file, or the name of a module in
    models/ of a checkout (or transitionlistener/models of a wheel). The folder of the file goes on ``sys.path``, so
    that it can import the modules beside it."""
    import importlib
    import importlib.util
    import sys

    model = str(model)
    if model.endswith(".py") or os.sep in model:
        folder = os.path.dirname(os.path.abspath(model))
        if folder not in sys.path:
            sys.path.insert(0, folder)
        return getattr(importlib.import_module(os.path.splitext(os.path.basename(model))[0]), potential)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for root in (os.path.join(here, "..", "..", "models"), os.path.join(here, "models")):
        if os.path.isdir(root):
            if os.path.abspath(root) not in sys.path:
                sys.path.insert(0, os.path.abspath(root))
            return getattr(importlib.import_module(model), potential)
    raise NLOGridError(f"no models directory next to {here}")


def _commit():
    """The git commit of this source tree, recorded in tables; None outside a checkout."""
    try:
        return subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def temporal_coupling(pot, p):
    """h3 (the paper's lambda_VL) of the temporal mass m_D^2 + h3 phi3^2 / 2 in GeV, for the matched parameters ``p``
    of the model ``pot``, as ``nucleationConf.temporal_coupling`` says: the matched value ``p["lambda_vl"]`` ("h3"),
    or None for its leading-order value 2 g3^2 ("2g3sq")."""
    choice = getattr(pot.config.nucleationConf, "temporal_coupling", "h3")
    if choice == "2g3sq":
        return None
    if choice != "h3":
        raise ValueError(f"nucleationConf.temporal_coupling is {choice!r}; it is 'h3' or '2g3sq'")
    if "lambda_vl" not in p:
        raise ValueError("temporal_coupling 'h3' needs the matched A0^2 phi^2 coupling, which this model's "
                         "soft_parameters does not supply as 'lambda_vl'; set temporal_coupling = '2g3sq'")
    return float(p["lambda_vl"])


def _h3_in_units(pot, p, unit):
    """:func:`temporal_coupling` with ``unit`` (in GeV) as the unit of energy."""
    h3 = temporal_coupling(pot, p)
    return None if h3 is None else h3 / unit


def _bounce(pot, T):
    """``(p, S3_over_T, r, phi3, info)`` at T (GeV): the matched parameters ``p`` and the bounce of V3
    (``abelian_higgs_3d.solve_LO_bounce``), with ``r`` in 1/GeV, ``phi3`` in GeV^(1/2) and ``info["dphi3"]`` in GeV^(3/2). The
    bounce is solved in units of T, since the shooting has absolute tolerances."""
    p = pot.soft_parameters(T)
    S3, r, phi3, info = abelian_higgs_3d.solve_LO_bounce(p["m3sq"] / T ** 2, p["lam3"] / T, p["g3sq"] / T, m_debye_sq=p["mD2"] / T ** 2,
                                              h3=_h3_in_units(pot, p, T))
    info = dict(info, dphi3=np.asarray(info["dphi3"]) * T ** 1.5)
    return p, S3, r / T, phi3 * math.sqrt(T), info


def _save(state, out):
    """Write ``state`` to the JSON file ``out`` atomically."""
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, out)


def _dim6(p, r, phi3, info):
    """The two parts of S_6 in eq. (A.2) on the bounce, ``(sextic, derivative)`` (``abelian_higgs_3d.dim6_action_shift``,
    ``abelian_higgs_3d.dim6_kinetic_action_shift``), or None when ``p`` has no dimension-six coefficients."""
    if "c_D2phi4" not in p:
        return None
    return (abelian_higgs_3d.dim6_action_shift(r, phi3, p["c6"]),
            abelian_higgs_3d.dim6_kinetic_action_shift(r, phi3, info["dphi3"], p["c_D2phi4"]))


def _in_units(p, r, phi3, unit):
    """The 3D theory with ``unit`` as the unit of energy: (r, phi3, m3sq, lam3, g3sq, mD2)."""
    return (r * unit, phi3 / math.sqrt(unit), p["m3sq"] / unit ** 2, p["lam3"] / unit, p["g3sq"] / unit,
            p["mD2"] / unit ** 2)


def fast_row(pot, T):
    """Every term of E, eq. (A.2), at T (GeV) but 1/2 Delta_mix and S_6, from ``abelian_higgs_3d.nlo_assembly`` in units of T.

    The row holds ``T_GeV``; ``S3_LO`` (S_3/T); ``determinants``, the sum of the one-loop terms (delta_h + delta_chi
    + delta_1 + delta_0 + S^{J=0}) minus C; ``radial`` (delta_h); ``log_A_dyn_over_T``; ``nlo_potential`` (the
    integral of V3^NLO); ``nlo_kinetic`` (for comparison only); ``total_no_mixing``, E without 1/2 Delta_mix; and,
    where the matching supplies the coefficients, the two parts of S_6, ``dS6`` and ``dSkin``.
    """
    p, S3, r, phi3, info = _bounce(pot, T)
    V3_GeV, Z3_GeV = pot.two_loop_potential(p), pot.two_loop_kinetic(p)
    r_u, phi_u, m3sq, lam3, g3sq, mD2 = _in_units(p, r, phi3, T)
    root = math.sqrt(T)
    V3n = lambda q: V3_GeV(q * root) / T ** 3
    Z3n = lambda q: Z3_GeV(q * root)
    a = abelian_higgs_3d.nlo_assembly(r_u, phi_u, m3sq, lam3, g3sq, mD2, S3, 0.0, T=1.0, V3_nlo=V3n, Z3_nlo=Z3n,
                           h3=_h3_in_units(pot, p, T))
    row = dict(T_GeV=T, S3_LO=float(S3),
               determinants=a["total_without_NLO_potential"] - S3,
               radial=a["radial"], log_A_dyn_over_T=a["log_A_dyn_over_T"],
               nlo_potential=a["nlo_potential"],
               nlo_kinetic=a["nlo_kinetic"],
               total_no_mixing=a["total"])
    dim6 = _dim6(p, r, phi3, info)
    if dim6 is not None:
        row["dS6"], row["dSkin"] = dim6
    return row


def mixing_row(pot, T, l_max):
    """Delta_mix of eq. (A.2) at T (GeV), ``abelian_higgs_3d.mixing_log_det`` summed to ``l_max`` in units of T, without its
    per-wave lists."""
    p, _, r, phi3, _ = _bounce(pot, T)
    r_u, phi_u, m3sq, lam3, g3sq, mD2 = _in_units(p, r, phi3, T)
    m = abelian_higgs_3d.mixing_log_det(r_u, phi_u, m3sq, lam3, g3sq, m_debye_sq=mD2, l_max=l_max, h3=_h3_in_units(pot, p, T))
    m.pop("log_t_on")
    m.pop("log_t_off")
    return m


def delta_S_of_row(row):
    """Delta S of eq. (A.4) for a row of :func:`fast_row` with its ``mixing`` (:func:`mixing_row`): E = total_no_mixing
    + 1/2 Delta_mix. S_6 is not included."""
    return abelian_higgs_3d.action_seam_correction(row["S3_LO"],
                                        row["total_no_mixing"] - row["S3_LO"] + 0.5 * row["mixing"]["value"],
                                        log_A_dyn_over_T=row["log_A_dyn_over_T"])


def build_table(pot, temperatures, out, *, l_max=abelian_higgs_3d.MIXING_L_MAX, metadata=None, log=None):
    r"""Build or resume the table of ``pot`` (a 3D model instance, configured) in the JSON file ``out``.

    ``temperatures`` is a list of ``(label, T_GeV, extra)``: ``label`` the key of the row, ``extra`` a dict stored in
    it. ``metadata`` is merged into the file. Fast rows are done for all temperatures before any mixing
    starts, so that partial output appears early. A file of another assembly version is refused. Returns the state as
    written.
    """
    log = log or (lambda msg: None)
    state = {}
    if os.path.exists(out):
        with open(out) as f:
            state = json.load(f)
    if state.get("assembly", ASSEMBLY) != ASSEMBLY:
        raise NLOGridError(f"{out} holds assembly {state['assembly']}, this code writes {ASSEMBLY}")
    state.update(metadata or {})
    state.update(l_max=l_max, matching="NLO", assembly=ASSEMBLY)
    rows = state.setdefault("rows", {})

    for label, T, extra in temperatures:
        old = rows.get(label, {})
        if old.get("error") or "total_no_mixing" in old:
            continue
        t0 = time.time()
        try:
            row = dict(extra)
            row.update(fast_row(pot, T))
            row["seconds"] = time.time() - t0
            rows[label] = row
            log(f"{label:<8s} S3/T={row['S3_LO']:9.3f}  total(no mixing)={row['total_no_mixing']:9.3f}")
        except errors.Timeout:
            raise
        except Exception as exc:
            rows[label] = dict(extra, T_GeV=T, error=f"{type(exc).__name__}: {exc}")
            log(f"{label:<8s} FAILED {type(exc).__name__}: {str(exc)[:80]}")
        _save(state, out)

    for label, T, extra in temperatures:
        row = rows.get(label, {})
        if "error" in row or "mixing" in row:
            continue
        t0 = time.time()
        try:
            row["mixing"] = mixing_row(pot, row["T_GeV"], l_max)
            row["mixing_seconds"] = time.time() - t0
            log(f"{label:<8s} mixing={row['mixing']['value']:9.3f}  spread={row['mixing']['spread']:.3f}  "
                f"({row['mixing_seconds']:.0f} s)")
        except errors.Timeout:
            raise
        except Exception as exc:
            row["mixing_error"] = f"{type(exc).__name__}: {exc}"
            log(f"{label:<8s} MIXING FAILED {type(exc).__name__}: {str(exc)[:80]}")
        _save(state, out)

    # every row that has the mixing enters the table, including rows added by an earlier or later call
    good = sorted((r for r in rows.values() if "mixing" in r), key=lambda r: r["T_GeV"])
    if len(good) >= 4:
        T = [row["T_GeV"] for row in good]
        dS = [delta_S_of_row(row) for row in good]
        S3 = [row["S3_LO"] for row in good]
        corr = abelian_higgs_3d.NLOCorrection(T, dS, S3_LO=S3)
        state["table"] = dict(T_GeV=list(map(float, corr.T_GeV)), delta_S=list(map(float, corr.delta_S)),
                              S3_LO=list(map(float, corr.S3_LO)),
                              mixing_uncertainty=0.5 * max(r_["mixing"]["spread"] for r_ in good))
        if all("dS6" in r_ for r_ in good):
            state["table"].update(dS6=[r_["dS6"] for r_ in good], dSkin=[r_["dSkin"] for r_ in good])
        state["T_GeV"], state["delta_S"] = state["table"]["T_GeV"], state["table"]["delta_S"]
        state["S3_LO"] = state["table"]["S3_LO"]
        state.pop("table_error", None)
        log(f"table: {corr}")
    else:
        state["table"] = None
        state["table_error"] = f"only {len(good)} temperatures have the mixing; a cubic needs 4"
        log(state["table_error"])
    _save(state, out)
    return state


def temperature_at_action(pot, S_target, ln_T_over_mu0=(3.0, -170.0), step=0.5, tol=1e-9, memo=None):
    r"""The temperature in GeV at which the leading-order action S_3/T of ``pot`` equals ``S_target``.

    S_3/T falls monotonically from the barrier edge (where it diverges) as T falls, down to the temperature where the
    barrier or the matching ends. The scan runs down in steps of ``step``
    in ln T, from above the barrier edge, on the scale mu0 of the couplings rather than on v: near y^4 = 3 g^4 the ratio
    v/mu0 is exponentially far from one. The root is then bisected, a temperature with no bounce counting as above the
    target on the hot side of the bracket. ``memo``, a dict, keeps the scan for the next target of the same table.
    """
    from scipy.optimize import brentq

    mu0 = pot.input_scale_GeV()
    memo = {} if memo is None else memo

    def action(x):
        if x not in memo:
            try:
                memo[x] = float(_bounce(pot, mu0 * math.exp(x))[1])
            except NO_BOUNCE:
                memo[x] = None
        return memo[x]

    above, seen_bounce, x = None, False, ln_T_over_mu0[0]
    while x >= ln_T_over_mu0[1]:
        s = action(x)
        if s is None and seen_bounce:
            break                       # below the coldest bounce with the action still above the target
        if s is not None and s < S_target:
            if above is None:           # already below at the hottest temperature tried: no bracket
                break
            return mu0 * math.exp(brentq(lambda q: _above(action(q), S_target), x, above, xtol=tol, rtol=tol))
        seen_bounce = seen_bounce or s is not None
        above = x
        x -= step
    raise NLOGridError(f"no temperature with S_3/T = {S_target:g} found for ln(T/mu0) in {tuple(ln_T_over_mu0)}")


def _above(s, target):
    """The function bisected: inside the bracket, a temperature with no bounce lies above the barrier edge."""
    return 1e6 if s is None else s - target


def build_node(pot, S_targets, out_stem, *, mu4_factors=(1.0,), l_max=abelian_higgs_3d.MIXING_L_MAX, metadata=None, log=None):
    """One grid node: for each factor of mu4/T, the table at the temperatures where S_LO = S_targets.

    Files ``<out_stem>_mu<factor>.json``. ``nucleationConf.mu4_over_T`` of ``pot`` is the central value, and is restored.
    A temperature found by an earlier run is taken from the file, so that a resumed node does not move its rows.
    Returns the list of files.
    """
    conf = pot.config.nucleationConf
    central = conf.mu4_over_T
    files = []
    try:
        for factor in mu4_factors:
            conf.mu4_over_T = central * factor
            out = f"{out_stem}_mu{factor:g}.json"
            known = {}
            if os.path.exists(out):
                with open(out) as f:
                    known = json.load(f).get("rows", {})
            temperatures, memo = [], {}
            for s in sorted(float(q) for q in S_targets):
                label = f"S{s:g}"
                try:
                    T = known[label]["T_GeV"] if label in known else temperature_at_action(pot, s, memo=memo)
                except NLOGridError as exc:
                    known_row = dict(target_S=s, T_GeV=None, error=str(exc))
                    if log:
                        log(f"{label:<8s} {exc}")
                    _record_failed(out, label, known_row)
                    continue
                temperatures.append((label, T, dict(target_S=s)))
            meta = dict(metadata or {}, mu4_over_T=conf.mu4_over_T, mu4_over_T_central=central, mu4_factor=factor,
                        code_commit=_commit())
            build_table(pot, temperatures, out, l_max=l_max, metadata=meta, log=log)
            files.append(out)
    finally:
        conf.mu4_over_T = central
    return files


def _record_failed(out, label, row):
    """Store the row ``label`` of a temperature that could not be found in the table file ``out``."""
    state = {}
    if os.path.exists(out):
        with open(out) as f:
            state = json.load(f)
    state.setdefault("rows", {})[label] = row
    _save(state, out)


def model_metadata(pot, model_name):
    """The fields of a node table that say which theory and which conventions it belongs to."""
    conf = pot.config.nucleationConf
    meta = dict(model=model_name, v_GeV=pot.v_GeV, mu0_GeV=pot.input_scale_GeV(), mu3_choice=getattr(conf, "mu3_choice", "g3sq"),
                input_scheme=getattr(conf, "input_scheme", "CW"), matching_order=conf.matching_order,
                temporal_coupling=getattr(conf, "temporal_coupling", "h3"), g3sq_order=conf.matching_order)
    meta.update(pot.nlo_grid_point())
    meta["couplings"] = list(pot.nlo_grid_point())
    return meta


# ----------------------------------------------------------------------
# the grid file
# ----------------------------------------------------------------------
def _node_arrays(path, S_nodes):
    """r, the shifts, ln(T/mu0) and the half spread of one node table on S_nodes, with its coldest S_LO.

    A node has to reach the hottest S_LO; below its coldest one, where the search found no bounce, the values are held, for the
    sake of the interpolant, and S_floor marks them."""
    with open(path) as f:
        d = json.load(f)
    if d.get("assembly") != ASSEMBLY:
        raise NLOGridError(f"{path} holds assembly {d.get('assembly')}, this code reads {ASSEMBLY}")
    rows = sorted((r for r in d["rows"].values() if "mixing" in r and "error" not in r), key=lambda r: r["S3_LO"])
    S = np.array([r["S3_LO"] for r in rows])
    if S.size < 2 or not bool(S[-1] >= S_nodes[-1] * (1 - 1e-7)):
        raise NLOGridError(f"{path}: the table covers S_LO up to {S[-1] if S.size else float('nan'):g}, "
                           f"not {S_nodes[-1]:g}")
    if not bool(S[0] < S_nodes[-1]) or np.any(np.diff(S) <= 0.0):
        raise NLOGridError(f"{path}: S_LO is not monotonic in the table")
    for key, r in d["rows"].items():    # only the cold end of a table may stop short
        target = r.get("target_S", r.get("S3_LO"))
        unfinished = "target_S" in r and "mixing" not in r
        if ("error" in r or "mixing_error" in r or unfinished) and target is not None and target > S[0]:
            raise NLOGridError(f"{path}: row {key} failed inside the table: {r.get('error') or r.get('mixing_error')}")
    x = np.log(S)
    q = np.clip(np.log(S_nodes), x[0], x[-1])
    dS = np.array([delta_S_of_row(r) for r in rows])

    def on_nodes(values):
        return PchipInterpolator(x, np.asarray(values, dtype=float))(q)

    dim6 = all("dS6" in r for r in rows)
    return dict(numbers={k: v for k, v in d.items() if isinstance(v, (int, float))}, couplings=d.get("couplings"),
                r=on_nodes(dS / S),
                r_dS6=on_nodes([r["dS6"] / r["S3_LO"] for r in rows]) if dim6 else np.full(q.size, np.nan),
                r_dSkin=on_nodes([r["dSkin"] / r["S3_LO"] for r in rows]) if dim6 else np.full(q.size, np.nan),
                ln_T_over_mu0=on_nodes([math.log(r["T_GeV"] / d["mu0_GeV"]) for r in rows]),
                half_spread=on_nodes([0.5 * r["mixing"]["spread"] for r in rows]),
                S_floor=float(S_nodes[S_nodes >= S[0] * (1.0 - 1e-7)][0]),
                header=dict({k: d.get(k) for k in ("model", "mu3_choice", "input_scheme", "matching_order", "l_max",
                                                   "assembly", "mu4_factor")},
                            **{k: d.get(k, CONVENTION_UNRECORDED[k]) for k in CONVENTION_UNRECORDED}),
                mu4_over_T=d["mu4_over_T"], central=d.get("mu4_over_T_central", d["mu4_over_T"]),
                code_commit=d.get("code_commit"))


def assemble_grid(node_tables, S_nodes, out, *, purpose, couplings=None, extra_header=None):
    r"""Write the grid file ``out`` (npz) from node tables.

    ``node_tables`` is a dict ``{mu4_factor: [json files]}``, the same coupling nodes in the same order for every
    factor; ``S_nodes`` the common values of S_LO, ascending; ``purpose`` "production" or "example" (the name a loader
    shows); ``couplings`` the names of the couplings the grid is a function of, by default those the node tables record
    (the keys of the model's ``nlo_grid_point``). The conventions go into the header from the
    tables, which have to agree; the header also gets the leave-one-out interpolation error of each factor.
    """
    S_nodes = np.array(sorted(float(s) for s in S_nodes))
    factors = sorted(float(f) for f in node_tables)
    arrays = {f: [_node_arrays(p, S_nodes) for p in node_tables[f]] for f in factors}
    first = arrays[factors[0]][0]
    couplings = list(couplings or first["couplings"] or [])
    if not couplings:
        raise NLOGridError("the node tables do not record their couplings; name them (couplings=...)")
    for f in factors:
        for n in arrays[f]:
            missing = [c for c in couplings if c not in n["numbers"]]
            if missing:
                raise NLOGridError(f"a node table has no value of {missing}")
            n["point"] = {c: n["numbers"][c] for c in couplings}
    head = {k: v for k, v in first["header"].items() if k != "mu4_factor"}
    for f in factors:
        for n in arrays[f]:
            different = {k: (n["header"].get(k), head[k]) for k in head if n["header"].get(k) != head[k]}
            if different:
                raise NLOGridError(f"node {n['point']} differs from the first in {different}")
            if not math.isclose(n["central"], first["central"], rel_tol=1e-12):
                raise NLOGridError(f"node {n['point']}: mu4/T {n['central']} against {first['central']}")
    points = np.array([[n["point"][c] for c in couplings] for n in arrays[factors[0]]], dtype=float)
    for f in factors:
        if not np.array_equal(points, np.array([[n["point"][c] for c in couplings] for n in arrays[f]], dtype=float)):
            raise NLOGridError(f"the nodes of mu4 factor {f:g} are not those of factor {factors[0]:g}")

    def stack(key):
        return np.array([[n[key] for n in arrays[f]] for f in factors])

    header = dict(head, purpose=purpose, couplings=list(couplings), mu4_over_T=first["central"], mu4_factors=factors,
                  profile_end_fraction=abelian_higgs_3d.TAIL_END_FRACTION, n_nodes=len(points), variable="S_LO",
                  quantity="Delta S / S_LO", rate_margin=RATE_MARGIN,
                  code_commits=sorted({n["code_commit"] for f in factors for n in arrays[f] if n["code_commit"]}))
    header.update(extra_header or {})
    data = dict(points=points, S_nodes=S_nodes, r=stack("r"), r_dS6=stack("r_dS6"), r_dSkin=stack("r_dSkin"),
                ln_T_over_mu0=stack("ln_T_over_mu0"), half_spread=stack("half_spread"), S_floor=stack("S_floor"))
    header["leave_one_out"] = {f"{f:g}": leave_one_out(points, S_nodes, data["r"][i], data["ln_T_over_mu0"][i],
                                                       data["S_floor"][i]) for i, f in enumerate(factors)}
    write_grid(out, header, data)
    return header


def write_grid(out, header, data):
    """Write the grid file ``out`` (npz): the JSON ``header`` and the arrays ``data``, atomically."""
    if not (os.path.dirname(out) == "" or os.path.isdir(os.path.dirname(out))):
        os.makedirs(os.path.dirname(out))
    tmp = out + ".tmp.npz"
    np.savez_compressed(tmp, header=np.array(json.dumps(header)), **data)
    os.replace(tmp, out)


def _interpolant(points, values):
    """values[node, ...] as a function of the couplings; returns nan outside the nodes' hull."""
    if points.shape[1] == 1:
        g = points[:, 0]
        o = np.argsort(g)
        spline = CubicSpline(g[o], values[o], axis=0)

        def one(q):
            q = float(q[0])
            if not bool(g.min() <= q <= g.max()):
                return np.full(values.shape[1:], np.nan)
            return spline(q)
        return one
    tri = CloughTocher2DInterpolator(points, values)
    return lambda q: tri(np.array([q], dtype=float))[0]


def leave_one_out(points, S_nodes, r, ln_T_over_mu0, S_floor=None):
    r"""The error of the interpolation at every node, from the grid without it, in the exponent and in T.

    Returns the median, 90 % and largest ``|delta(Delta S)| = S_LO |r_interpolated - r|``, the same for the implied
    ``|delta ln T| = |delta(Delta S)| / |d S_LO/d ln T|``, over the nodes that lie inside the hull of the others and the
    values of S_LO their tables hold, the number of those nodes, and the worst node. A held row of another node enters
    through the gradients of the interpolant, so that a grid with floors reports more than its error.
    """
    d_lnT_dS = np.gradient(ln_T_over_mu0, S_nodes, axis=1)
    errors_S, errors_T, worst, inside = [], [], (None, -1.0), 0
    for i in range(len(points)):
        keep = np.arange(len(points)) != i
        try:
            pred = _interpolant(points[keep], r[keep])(points[i])
        except QhullError:      # the other nodes are collinear
            continue
        if not np.all(np.isfinite(pred)):
            continue
        inside += 1
        held = S_nodes >= (S_nodes[0] if S_floor is None else S_floor[i])
        e = (np.abs(pred - r[i]) * S_nodes)[held]
        errors_S.append(e)
        errors_T.append(e * np.abs(d_lnT_dS[i][held]))
        if float(e.max()) > worst[1]:
            worst = (points[i].tolist(), float(e.max()))
    if not errors_S:
        return dict(nodes_inside=0)
    es, et = np.concatenate(errors_S), np.concatenate(errors_T)
    return dict(nodes_inside=inside, exponent_median=float(np.median(es)), exponent_90=float(np.quantile(es, 0.9)),
                exponent_max=float(es.max()), ln_T_median=float(np.median(et)), ln_T_90=float(np.quantile(et, 0.9)),
                ln_T_max=float(et.max()), worst_node=worst[0], worst_node_exponent=worst[1])


def _with_dim6(S_nodes, r, r6, tol=1e-12, max_iter=100):
    """Delta S / S_LO on the nodes with S_6 inside the exponent E of eq. (A.2), from the grid's r = Delta S_0 / S_LO
    without it and r6 = S_6 / S_LO.

    Delta S_0 solves eq. (A.4) for E without S_6. Adding S_6 to E changes eq. (A.4) to
    Delta S = Delta S_0 + S_6 + (3/2) ln[(S_LO + Delta S) / (S_LO + Delta S_0)], solved here by fixed-point iteration
    (the map contracts by (3/2) / (S_LO + Delta S) << 1). Adding S_6 to Delta S alone would leave it outside the
    logarithm, which is correct only to first order in the operators."""
    S = np.asarray(S_nodes, dtype=float)
    d0, d6 = np.asarray(r, dtype=float) * S, np.asarray(r6, dtype=float) * S
    d = d0 + d6
    for _ in range(max_iter):
        new = d0 + d6 + 1.5 * np.log((S + d) / (S + d0))
        if np.all(np.abs(new - d) <= tol * np.maximum(1.0, np.abs(new))):
            return new / S
        d = new
    raise RuntimeError("the dimension-six fixed point of eq. (A.4) did not converge")


class NLOGrid:
    """A grid file of :func:`assemble_grid` in memory; :meth:`correction` makes the object a model evaluates."""

    def __init__(self, path):
        self.path = path
        with np.load(path, allow_pickle=False) as f:
            self.header = json.loads(str(f["header"]))
            self.points, self.S_nodes = f["points"], f["S_nodes"]
            self.r, self.r_dS6, self.r_dSkin = f["r"], f["r_dS6"], f["r_dSkin"]
            self.ln_T_over_mu0, self.half_spread = f["ln_T_over_mu0"], f["half_spread"]
            self.S_floor = f["S_floor"]
        self.couplings = tuple(self.header["couplings"])
        self.factors = [float(q) for q in self.header["mu4_factors"]]
        self._cache = {}
        self._tri = None

    def _factor_index(self, factor):
        for i, f in enumerate(self.factors):
            if math.isclose(f, factor, rel_tol=1e-9):
                return i
        raise NLOGridError(f"{self.describe()}: mu4 factor {factor:g} is not in the grid, which has {self.factors}")

    def describe(self):
        """The grid's name for messages."""
        return f"NLO grid {os.path.basename(self.path)} ({self.header['purpose']})"

    def coupling_range(self):
        """The range of each coupling of the nodes, for messages."""
        return ", ".join(f"{c} in [{self.points[:, i].min():g}, {self.points[:, i].max():g}]"
                         for i, c in enumerate(self.couplings))

    def _at(self, name, values, i, q):
        """``values[i]`` interpolated to the couplings ``q``; the interpolant is built once per name and factor."""
        key = (name, i)
        if key not in self._cache:
            self._cache[key] = _interpolant(self.points, values[i])
        return self._cache[key](q)

    def correction(self, point, mu4_factor=1.0, *, dim6=False, hot_side_negligible=False, cold_side_negligible=False):
        """The :class:`GridCorrection` at the coupling ``point`` (a dict with the grid's couplings) for the factor
        ``mu4_factor`` of the matching scale; with ``dim6`` S_6 is part of E in eq. (A.2) (:func:`_with_dim6`)."""
        missing = [c for c in self.couplings if c not in point]
        if missing:
            raise NLOGridError(f"{self.describe()} is a function of {self.couplings}; the point lacks {missing}")
        q = [float(point[c]) for c in self.couplings]
        i = self._factor_index(mu4_factor)
        r = self._at("r", self.r, i, q)
        if not np.all(np.isfinite(r)):
            raise NLOGridError(f"the coupling point {dict(zip(self.couplings, q))} is outside the {self.describe()}: "
                               f"{self.coupling_range()}, {len(self.points)} nodes"
                               + (" (and only inside their convex hull)" if len(self.couplings) > 1 else ""))
        if dim6:
            if not np.all(np.isfinite(self.r_dS6)):
                raise NLOGridError(f"{self.describe()} stores no dimension-six shift")
            r = _with_dim6(self.S_nodes, r, self._at("dS6", self.r_dS6, i, q) + self._at("dSkin", self.r_dSkin, i, q))
        return GridCorrection(self.S_nodes, r, self._at("x", self.ln_T_over_mu0, i, q), dict(zip(self.couplings, q)),
                              self.describe(), hot_side_negligible, cold_side_negligible, self._floor(i, q))

    def _floor(self, i, q):
        """The lowest S_LO held at q: the largest of the floors of the nodes q is interpolated from."""
        floors = self.S_floor[i]
        if len(self.couplings) == 1:
            g = self.points[:, 0]
            lo, hi = g[g <= q[0]].max(), g[g >= q[0]].min()
            return float(floors[(g == lo) | (g == hi)].max())
        if self._tri is None:
            self._tri = Delaunay(self.points)
        simplex = self._tri.find_simplex(np.array([q], dtype=float))[0]
        return float(floors[self._tri.simplices[simplex]].max())


_GRIDS = {}


def load_grid(path):
    """The grid in ``path``; read once per process."""
    key = os.path.abspath(path)
    if key not in _GRIDS:
        if not os.path.exists(path):
            raise NLOGridError(f"no NLO grid at {path}: "
                               "attach a table to nlo_correction, or set nucleationConf.nlo_grid_file.")
        _GRIDS[key] = NLOGrid(path)
    return _GRIDS[key]


def _continued(r_edge, slope, u, width=0.5):
    """r_edge + slope width tanh(u/width): the slope of r at the edge, and bounded however far u = ln(S/S_edge) goes."""
    return r_edge + slope * width * np.tanh(u / width)


class GridCorrection:
    """Delta S of eq. (A.4) at one coupling point as a function of S_LO = S_3/T (see the module docstring).

    Called as ``correction(T_GeV, S3_LO=S)``, like ``abelian_higgs_3d.NLOCorrection``, by the model's ``actionNuc``; ``T_GeV``
    appears only in error messages. Inside ``S_range`` it is ``S_LO r(S_LO)``, with ``r`` a quintic spline in ln S_LO
    through the nodes from the lowest S_LO held (``S_floor``) up; outside, see :func:`_continued` and
    ``hot_side_negligible``, ``cold_side_negligible``. r is smooth in S_LO, and a spline of high order follows its
    curvature between the nodes, where a monotone interpolant (PCHIP) misses it by up to 0.04 in Delta S.
    """

    def __init__(self, S_nodes, r, ln_T_over_mu0, point, source, hot_side_negligible, cold_side_negligible, S_floor=None):
        lo = float(S_nodes[0]) if S_floor is None else float(S_floor)
        self.S_range = (lo, float(S_nodes[-1]))
        self._x = np.log(S_nodes)
        self.r_nodes = np.array(r, dtype=float)
        held = self._x >= math.log(lo) - 1e-12           # below S_floor the nodes hold copies of the edge value
        self._r = make_interp_spline(self._x[held], self.r_nodes[held], k=min(5, int(held.sum()) - 1))
        ends = np.log(self.S_range)
        self.r_edges = self._r(ends)
        self._slopes = self._r.derivative()(ends)
        self.ln_T_over_mu0 = tuple(float(q) for q in np.interp(ends, self._x, ln_T_over_mu0))
        self.point, self.source = point, source
        self.hot_side_negligible, self.cold_side_negligible = bool(hot_side_negligible), bool(cold_side_negligible)
        self.continued_calls = 0

    def _refuse(self, T_GeV, S, side):
        """Raise :class:`NLOGridError` for a query outside the range of S_LO."""
        raise NLOGridError(
            f"{self.source} at {self.point}: T = {float(T_GeV):.6g} GeV has S_LO = {float(S):.6g}, {side} the range "
            f"[{self.S_range[0]:g}, {self.S_range[1]:g}] of S_LO, and the rate is not known to be irrelevant there "
            f"(ln(T/mu0) of the range: [{self.ln_T_over_mu0[0]:.4g}, {self.ln_T_over_mu0[1]:.4g}]). "
            "Nothing is extrapolated.")

    def __call__(self, T_GeV, S3_LO=None):
        """Delta S at the leading-order action ``S3_LO``."""
        if S3_LO is None:
            raise NLOGridError("a grid correction is a function of the leading-order action: pass S3_LO")
        S = float(S3_LO)
        if not S > 0.0:
            self._refuse(T_GeV, S, "below")
        lo, hi = self.S_range
        if S < lo:
            if not self.cold_side_negligible:
                self._refuse(T_GeV, S, "below")
            self.continued_calls += 1
            return float(S * _continued(self.r_edges[0], self._slopes[0], math.log(S / lo)))
        if S > hi:
            if not self.hot_side_negligible:
                self._refuse(T_GeV, S, "above")
            self.continued_calls += 1
            return float(S * _continued(self.r_edges[1], self._slopes[1], math.log(S / hi)))
        return float(S * self._r(math.log(S)))

    def __repr__(self):
        return f"GridCorrection({self.point}, S_LO in [{self.S_range[0]:g}, {self.S_range[1]:g}], {self.source})"


def _check_conventions(grid, conf):
    """Refuses a configuration that differs from the grid's; returns the factor of the configured mu4/T."""
    head = grid.header
    for key, attr in CONVENTIONS.items():
        have = getattr(conf, attr, None)
        if have != head.get(key, CONVENTION_UNRECORDED.get(key)):
            built = head.get(key, CONVENTION_UNRECORDED.get(key))
            if key == "g3sq_order":
                raise NLOGridError(f"{grid.describe()} was built with g3^2 at {built}; this run matches g3^2 at "
                                   f"{have} (matching_order). Attach a table built with this version.")
            raise NLOGridError(f"{grid.describe()} was built with {key} = {built!r}; this run has {attr} = {have!r}. "
                               f"Set nucleationConf.{attr} = {built!r} or attach a table built for your settings.")
    factor = conf.mu4_over_T / head["mu4_over_T"]
    if not any(math.isclose(factor, f, rel_tol=1e-9) for f in grid.factors):
        raise NLOGridError(f"{grid.describe()} holds mu4/T = {[head['mu4_over_T'] * f for f in grid.factors]}; "
                           f"this run has mu4_over_T = {conf.mu4_over_T}. Set one of those, or attach a table "
                           "built for your settings.")
    return factor


def correction_for_model(pot):
    """The :class:`GridCorrection`, Delta S of eq. (A.4) as a function of S_3/T, of the model ``pot`` at its couplings:
    what a model with ``include_determinants`` and no attached table uses.

    The grid is ``nucleationConf.nlo_grid_file`` or the shipped ``tab_data/nlo_grids/<pot.nlo_grid_name>.npz``; with
    ``nucleationConf.include_dim6`` S_6 is included. Raises :class:`NLOGridError` if the file is absent, its
    conventions differ from the run's, or the point is outside it. Queries outside the grid's range of S_LO are allowed
    only as the module docstring says, decided here at the model's v with the Hubble rate of its vacuum energy.
    """
    conf = pot.config.nucleationConf
    path = getattr(conf, "nlo_grid_file", None) or os.path.join(GRID_DIR, f"{pot.nlo_grid_name}.npz")
    grid = load_grid(path)
    if grid.header.get("model") != pot.nlo_grid_name:
        raise NLOGridError(f"{grid.describe()} is for the model {grid.header.get('model')!r}, not {pot.nlo_grid_name!r}")
    factor = _check_conventions(grid, conf)
    dim6 = bool(getattr(conf, "include_dim6", False))
    corr = grid.correction(pot.nlo_grid_point(), factor, dim6=dim6)

    # ln(Gamma/H^4) at the two ends of the range, with the rate of actionNuc and H from the vacuum energy alone, the
    # smallest H there can be: an upper bound of the rate at the hot end. At the cold end radiation is negligible.
    ln_H = 0.5 * math.log(pot.vacuum_energy_GeV4() / 3.0) - math.log(M_PL_GEV)
    ln_mu0 = math.log(pot.input_scale_GeV())

    def ln_rate_over_H4(S_LO, r, ln_T_over_mu0):
        S = S_LO * (1.0 + r)
        return 4.0 * (ln_mu0 + ln_T_over_mu0 - ln_H) + 1.5 * math.log(S / (2.0 * math.pi)) - S

    lo, hi = corr.S_range
    corr.hot_side_negligible = bool(ln_rate_over_H4(hi, corr.r_edges[1], corr.ln_T_over_mu0[1]) < -RATE_MARGIN)
    corr.cold_side_negligible = bool(ln_rate_over_H4(lo, corr.r_edges[0], corr.ln_T_over_mu0[0]) > RATE_MARGIN)
    # a model can name couplings above which its NLO expansion, not the grid, limits the accuracy
    point = pot.nlo_grid_point()
    for name, limit in getattr(pot, "nlo_expansion_limits", {}).items():
        if point.get(name, -math.inf) > limit:
            warnings.warn(f"{name} > {limit}: the half-range of the NLO exponent under mu3 x 0.5, 2 exceeds 2 % at "
                          "S_3/T = 150, so that the first-order Delta S is not reliable")
    return corr
