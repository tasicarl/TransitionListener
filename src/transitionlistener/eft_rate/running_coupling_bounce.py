"""The four-dimensional rate with the couplings run to a temperature-dependent scale: the bounce of a model's potential
``V_rg``, the action as a table, and the table's construction.

A model mixes in ``RunningCouplingBounce`` and supplies ``V_rg(X, T, include_radiation)``, its daisy-resummed
one-loop potential with the couplings run from the input scale ``mu0`` to a scale proportional to T. Its phases, vacuum
energy and thermodynamics stay those of its parent, the four-dimensional model with fixed couplings, because at the
broken minimum, far above T, the logarithms of the running are large. Only the bounce is solved on ``V_rg``, with the
false vacuum at the origin, and its action reaches the pipeline through ``bounceFreeAction``. The rate is the
four-dimensional one, T^4 (S_3/2 pi T)^{3/2} exp(-S_3/T).

At fixed couplings S_3/T depends on T/mu0 only, so one table per coupling point serves every vacuum expectation
value. A model that names a table (``rg_table_name``, ``rg_table_point()``, ``rg_reference_action(ln_x)``) reads
S_3/T from the table in ``tab_data/rg_action_tables`` wherever its couplings lie inside the table and its scale factor
``mu_over_piT`` is the table's; elsewhere every temperature is solved directly, at about ten seconds per bounce. The
table holds, on fixed values of S_3/T (``S_NODES``), ln(T/mu0) minus the same quantity of a reference action that
the model computes cheaply (for the conformal U(1), the leading-order action of its 3D theory). That difference is
smooth in the couplings; it is interpolated in them as the NLO tables are (``nlo_tables._interpolant``), and the
reference is added back at the couplings of the run.

    python -m transitionlistener.eft_rate.running_coupling_bounce build --model-file models/TL_conformal_u1_fermions_rg.py \\
        --param g=0.5 --param y=0.3 --out nodes/g0.5_y0.3.json
    python -m transitionlistener.eft_rate.running_coupling_bounce assemble --model-file models/TL_conformal_u1_fermions_rg.py \
        --nodes "nodes/*.json" --out table.npz

Documentation: https://tasillo.de/TransitionListener/

Authors:
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np

#: the wall of the bounce solver at this multiple of the escape point, its tolerances, and the step of the derivative
#: in units of the escape point
BOUNCE_WALL, BOUNCE_TOL, BOUNCE_STEP = 30.0, 1e-8, 1e-4
#: S_3/T where the potential has no barrier at the origin or the bounce fails: no nucleation
ACTION_CAP = 1.2e3
#: the values of S_3/T of a table: below the lowest, 15, the rate exceeds H^4 by far (a transition needs about
#: 4 ln(T/H_V), at least 50 for the temperatures of interest), and above the highest, exp(-500), it is
#: negligible at every temperature above 1e-15 GeV, where a transition needs S_3/T of at most about 4 ln(M_Pl/T) = 310
S_NODES = tuple(float(s) for s in np.geomspace(15.0, 500.0, 41))
#: temperatures per decade of T/mu0 at which a node is solved, and the range of T/mu0 it walks
PER_DECADE, X_LOWEST, X_HIGHEST = 8, 1e-40, 1e2
TABLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tab_data", "rg_action_tables")


def solve_action_over_T(pot, T):
    """S_3/T of the bounce on ``pot.V_rg`` at T (internal units): ``ACTION_CAP`` without a barrier, nan where the
    bounce solver fails."""
    from scipy.optimize import brentq
    from transitionlistener import errors
    from transitionlistener.eft_rate import abelian_higgs_3d
    if T <= 0.0:
        return ACTION_CAP
    V0 = float(np.ravel(pot.V_rg(np.array([[0.0]]), T, include_radiation=False))[0])

    def W(p):
        a = np.asarray(p, float)
        out = np.asarray(pot.V_rg(np.atleast_1d(a)[..., None], T, include_radiation=False),
                         float).reshape(np.atleast_1d(a).shape) - V0
        return float(out[0]) if a.ndim == 0 else out
    try:
        # W(phi) for phi << T is below the rounding of V(0) ~ T^4, so the search starts at 1e-2 T
        a = b = 1e-2 * T
        if not W(a) > 0:
            return ACTION_CAP
        while W(b) >= 0:
            b *= 1.3
            if b > 1e6 * T:                                   # no escape point: no barrier to tunnel through
                return ACTION_CAP
        esc = brentq(W, a, b, xtol=1e-14 * T, rtol=1e-14)
        h = BOUNCE_STEP * esc
        dW = lambda p: (W(p - 2 * h) - 8 * W(p - h) + 8 * W(p + h) - W(p + 2 * h)) / (12 * h)
        S = abelian_higgs_3d.solve_bounce_1d(W, dW, esc, bounded=False, wall_multiple=BOUNCE_WALL,
                                             xtol=BOUNCE_TOL, phitol=BOUNCE_TOL)[0]
    except ValueError:                                        # no barrier between the origin and the escape point
        return ACTION_CAP
    except (RuntimeError, ArithmeticError, errors.TunnelingError):
        return math.nan
    return min(S / T, ACTION_CAP) if np.isfinite(S) and S > 0 else math.nan


class RunningCouplingBounce:
    """The bounce on ``V_rg`` with the false vacuum at the origin, in place of the 4D bounce on ``Vtot``."""

    #: the name of the shipped table, tab_data/rg_action_tables/<name>.npz, or None to solve every temperature
    rg_table_name = None

    def rg_table_point(self) -> dict:
        """The couplings the table is a function of; supplied by a model that names a table."""
        raise NotImplementedError

    def rg_reference_action(self, ln_x):
        """A cheap reference S_3/T at T/mu0 = exp(ln_x), or None; supplied by a model that names a table."""
        raise NotImplementedError

    def bounceFreeAction(self):
        """The action S3 at T (internal units) from :meth:`action_over_T`."""
        return lambda T: float(T) * self.action_over_T(float(T))

    def action_over_T(self, T):
        """S_3/T at T (internal units): from the table where it applies, by a direct solve otherwise."""
        table = self._rg_table()
        if table is None:
            S = solve_action_over_T(self, T)
            return ACTION_CAP if math.isnan(S) else S
        return table(math.log(float(T) / self.mu0)) if T > 0 else ACTION_CAP

    def _rg_table(self):
        """S_3/T as a function of ln(T/mu0) at the model's couplings, or None where no table applies."""
        key = (self.rg_table_name, self.mu_over_piT, tuple(sorted(self.rg_table_point().items()))
               if self.rg_table_name else None)
        if getattr(self, "_rg_table_key", None) != key:
            self._rg_table_key, self._rg_table_fn = key, None
            path = os.path.join(TABLE_DIR, f"{self.rg_table_name}.npz") if self.rg_table_name else None
            if path and os.path.exists(path):
                self._rg_table_fn = table_at(path, self.rg_table_point(), self.mu_over_piT,
                                             reference_ln_x(self.rg_reference_action))
        return self._rg_table_fn


def reference_ln_x(action, S_nodes=S_NODES, ln_x_grid=None):
    """ln(T/mu0) of a reference action on S_nodes, from the monotonic part of ``action(ln_x)`` inverted.

    Without ``ln_x_grid`` the action is sampled from the hot end down to just below the lowest of S_nodes, with a step
    that keeps the change of S_3/T between samples near two: fine where the action rises steeply towards the edge of the
    barrier, coarse in its slow tail, and never at the cold temperatures where the running of the couplings fails."""
    from scipy.interpolate import PchipInterpolator

    def at(x):
        try:
            s = action(float(x))
        except Exception:                                    # noqa: BLE001 -- the reference fails there: no value
            s = None
        return np.nan if s is None or not np.isfinite(s) else float(s)

    if ln_x_grid is not None:
        samples = [(float(x), at(x)) for x in ln_x_grid]
    else:
        samples, x, misses = [], math.log(X_HIGHEST), 0
        while x > -150.0:
            s = at(x)
            samples.append((x, s))
            if np.isnan(s) or s >= ACTION_CAP:
                if any(np.isfinite(v) and v < ACTION_CAP for _, v in samples[:-1]):
                    misses += 1
                    if misses > 5:                           # below the reference's range: done
                        break
                x -= 0.5
                continue
            misses = 0
            if s < 0.8 * S_nodes[0]:
                break
            prev = [(a, v) for a, v in samples[:-1] if np.isfinite(v) and v < ACTION_CAP]
            slope = abs((prev[-1][1] - s) / (prev[-1][0] - x)) if prev else 0.0
            x -= float(np.clip(2.0 / slope, 0.01, 0.5)) if slope > 0 else 0.05
    pts = sorted((x, v) for x, v in samples if np.isfinite(v) and v < ACTION_CAP)
    x_ok, S_ok = [], []
    for x, v in pts:
        if not S_ok or v > S_ok[-1]:
            x_ok.append(x); S_ok.append(v)
    return PchipInterpolator(np.array(S_ok), np.array(x_ok), extrapolate=True)(np.asarray(S_nodes, float))


def table_at(path, point, mu_over_piT, ref_ln_x):
    """The function ln(T/mu0) -> S_3/T of the table at the couplings ``point``, or None outside the table;
    ``ref_ln_x`` is the reference's ln(T/mu0) on the table's S nodes at these couplings."""
    from scipy.interpolate import PchipInterpolator
    from transitionlistener.eft_rate.nlo_tables import _interpolant
    from transitionlistener.eft_rate.one_field_rate_3d import saturate_action
    with np.load(path) as f:
        head = json.loads(str(f["header"]))
        points, S_nodes, ln_x = f["points"], f["S_nodes"], f["ln_x"]
    if not math.isclose(float(mu_over_piT), head["mu_over_piT"], rel_tol=1e-12):
        return None
    q = np.array([point[c] for c in head["couplings"]], dtype=float)
    lx = np.asarray(_interpolant(points, ln_x)(q), float) + np.asarray(ref_ln_x, float)
    if not np.all(np.isfinite(lx)) or np.any(np.diff(lx) <= 0):
        return None
    inv = PchipInterpolator(lx, S_nodes)
    lo_slope = (S_nodes[1] - S_nodes[0]) / (lx[1] - lx[0])

    hi_slope = (S_nodes[-1] - S_nodes[-2]) / (lx[-1] - lx[-2])

    def S_of(ln_x):
        if ln_x > lx[-1]:           # hotter than the table: the slope continued, rolled over to the cap
            return saturate_action(float(S_nodes[-1] + hi_slope * (ln_x - lx[-1])))
        if ln_x < lx[0]:            # colder than the table: the rate is far above H^4, the slope continued
            return max(float(S_nodes[0] + lo_slope * (ln_x - lx[0])), 1e-3)
        return float(inv(ln_x))
    return S_of


def build_node(model, params, log=print):
    """S_3/T on a ladder of PER_DECADE temperatures per decade of T/mu0, walked down from the hottest temperature with a
    barrier until S_3/T falls below the lowest of S_NODES, then resampled as ln(T/mu0) on S_NODES."""
    pot = model(params, verbose=False)
    step = math.log(10.0) / PER_DECADE
    rows = []
    # the hottest temperature with a barrier and an action below the cap, from above
    ln_x = math.log(X_HIGHEST)
    while ln_x > math.log(X_LOWEST):
        S = solve_action_over_T(pot, pot.mu0 * math.exp(ln_x))
        rows.append((ln_x, S))
        if S < S_NODES[-1]:
            break
        ln_x -= 4 * step if S >= ACTION_CAP else step
    # the edge of the barrier is crossed in steps of four: refine above the first row below the top node
    top = ln_x
    for k in range(1, 16):
        S = solve_action_over_T(pot, pot.mu0 * math.exp(top + 0.25 * k * step))
        rows.append((top + 0.25 * k * step, S))
        if S >= S_NODES[-1]:
            break
    rows.sort(key=lambda r: -r[0])
    ln_x = top
    while ln_x > math.log(X_LOWEST) and not rows[-1][1] <= S_NODES[0]:      # a failed (nan) row does not stop the walk
        ln_x -= step
        rows.append((ln_x, solve_action_over_T(pot, pot.mu0 * math.exp(ln_x))))
        if len(rows) % 20 == 0:
            log(f"{params}: {len(rows)} bounces, T/mu0 = {math.exp(ln_x):.2e}, S/T = {rows[-1][1]:.2f}")
    return dict(params=params, couplings=list(pot.rg_table_point()), point=pot.rg_table_point(),
                mu_over_piT=pot.mu_over_piT, model=pot.rg_table_name, rows=rows)


def node_on_S(node, S_nodes=S_NODES):
    """ln(T/mu0) of one node on S_nodes: the monotonic part of its ladder, inverted; below its coldest S held by the slope
    of its two coldest rows."""
    from scipy.interpolate import PchipInterpolator
    # a temperature where the solver failed (nan) is left out
    rows = sorted((r for r in node["rows"] if np.isfinite(r[1]) and r[1] < ACTION_CAP), key=lambda r: r[0])
    x, S = [], []
    for lx, s in rows:                                        # S_3/T rises with T where the table is used
        if not S or s > S[-1]:
            x.append(lx); S.append(s)
    x, S = np.array(x), np.array(S)
    S_nodes = np.asarray(S_nodes, float)
    if S[-1] < S_nodes[-1]:
        raise ValueError(f"node {node['point']}: the ladder stops at S/T = {S[-1]:.1f}, below {S_nodes[-1]:g}")
    out = PchipInterpolator(S, x)(np.clip(S_nodes, S[0], None))
    below = S_nodes < S[0]
    out[below] = x[0] + (S_nodes[below] - S[0]) * (x[1] - x[0]) / (S[1] - S[0])
    return out


def assemble(paths, out, model):
    """The table file ``out`` from node files and the model class (for the reference action); returns its header with
    the leave-one-out error in S_3/T."""
    from transitionlistener.eft_rate.nlo_tables import _interpolant
    nodes = [json.load(open(p)) for p in paths]
    head = dict(model=nodes[0]["model"], couplings=nodes[0]["couplings"], mu_over_piT=nodes[0]["mu_over_piT"],
                quantity="ln(T/mu0) on S_3/T minus that of the model's reference action", per_decade=PER_DECADE)
    for n in nodes:
        if (n["model"], n["couplings"], n["mu_over_piT"]) != (head["model"], head["couplings"], head["mu_over_piT"]):
            raise ValueError(f"node {n['point']} differs from the first")
    points = np.array([[n["point"][c] for c in head["couplings"]] for n in nodes], dtype=float)
    full = np.array([node_on_S(n) for n in nodes])
    ref = np.array([reference_ln_x(model(dict(n["params"]), verbose=False).rg_reference_action) for n in nodes])
    ln_x = full - ref                                        # the table: ln(T/mu0) minus the reference's
    S_nodes = np.array(S_NODES)
    errors, worst = [], (None, 0.0)
    slope = np.gradient(S_nodes, axis=0)[None, :] / np.gradient(full, axis=1)       # dS/dln x per node
    for i in range(len(points)):
        keep = np.arange(len(points)) != i
        try:
            pred = np.asarray(_interpolant(points[keep], ln_x[keep])(points[i]), float)
        except Exception:                                    # noqa: BLE001 -- collinear rest: not inside
            continue
        if not np.all(np.isfinite(pred)):
            continue
        e = np.abs(pred - ln_x[i]) * np.abs(slope[i])       # the error in S_3/T at fixed T
        errors.append(e)
        if e.max() > worst[1]:
            worst = (points[i].tolist(), float(e.max()))
    if errors:
        e = np.concatenate(errors)
        head["leave_one_out_S"] = dict(nodes_inside=len(errors), median=float(np.median(e)),
                                       p90=float(np.quantile(e, 0.9)), max=float(e.max()), worst_node=worst[0])
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out, header=np.array(json.dumps(head)), points=points, S_nodes=S_nodes, ln_x=ln_x)
    return head


def main(argv=None):
    from transitionlistener.eft_rate.nlo_tables import load_model
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="the ladder of one coupling point")
    b.add_argument("--model-file", required=True)
    b.add_argument("--potential", default="specific_potential")
    b.add_argument("--param", action="append", default=[], metavar="NAME=VALUE")
    b.add_argument("--out", required=True)
    a = sub.add_parser("assemble", help="collect node files into a table")
    a.add_argument("--model-file", required=True)
    a.add_argument("--potential", default="specific_potential")
    a.add_argument("--nodes", required=True, nargs="+")
    a.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.command == "build":
        params = {k: float(v) for k, _, v in (p.partition("=") for p in args.param)}
        params.setdefault("v_GeV", 1.0)
        t0 = time.time()
        node = build_node(load_model(args.model_file, args.potential), params)
        node["seconds"] = time.time() - t0
        with open(args.out, "w") as f:
            json.dump(node, f)
        print(f"{params}: {len(node['rows'])} bounces in {node['seconds']:.0f} s")
        return 0
    paths = sorted(p for pat in args.nodes for p in glob.glob(pat))
    print(json.dumps(assemble(paths, args.out, load_model(args.model_file, args.potential)), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
