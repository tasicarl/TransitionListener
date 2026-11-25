"""LO bounce actions on the line-scan temperature grids, for the exact-percolation variants.

Tasks (g, kind):
  nlo1   S3 on the NLO-matched theory at mu4 = 7.0553 T, plus the 3D expansion parameter
         x3 = g3^2/(pi m3) at each T
  nlo05  S3 at mu4 x 0.5
  nlo2   S3 at mu4 x 2
  lom    S3 on the LO-matched theory
  mu3x05, mu3x2  S3 on the NLO-matched theory with the EFT scale mu3 = g T halved / doubled
Grid: T = geomspace(0.05, 8) x anchor (the grid of rate_history_u1.py) for the couplings of G. Parallel over tasks.
Writes <ACTION_CACHE>/act_<kind>_g<g>.json (paths.ACTION_CACHE, paper/data/action_cache unless ACTION_CACHE_DIR is
set); files that exist are skipped.
With --low: the extension of the grid below 0.05 x anchor, geomspace(1e-5, 0.05) x anchor with
200 points (endpoint excluded), for the couplings of LOW_G, where percolation at large v needs
temperatures below the main grid. Writes act_<kind>_g<g>_low.json. With --low-all the same
for every coupling of G; build_exact_table.py reads the _low files wherever they exist.
With --g g1,g2,... only these couplings, on both grids (used for the half steps of the slow tail).
With --high the extension above the grid, geomspace(8, 60) x anchor (40 points), for every coupling of
build_exact_table.G: at small v the transition percolates close to 8 x anchor, and the bubbles nucleated above
the main grid would be missing. Writes act_<kind>_g<g>_high.json.
With --higher the further extension geomspace(60, 2000) x anchor (40 points) for the couplings of HIGHER_G:
at g <= 0.425 and small v the transition percolates within HOT_MARGIN of 60 x anchor, so build_exact_table.py
would mark those nodes window-limited. Writes act_<kind>_g<g>_high2.json.
With --highest the extension geomspace(2000, 2e5) x anchor (40 points) for HIGHEST_G: the faster variants
(hord+, pref+, adynT) percolate even hotter at g <= 0.405. Writes act_<kind>_g<g>_high3.json.
With --fig1 the LO actions at mu4 x 0.5 and x 2 at g = 0.5 (KINDS_FIG1, main grid only), for the matching-scale
band of the rate figure.
With --edge every grid (main, --low, --high, --higher, --highest) for the couplings of EDGE_G, from g = 0.38 to 0.405,
where figure 1 follows the line to the end of percolation.
Every kind stores, on its own LO bounce, the first-order shifts of the two dimension-six operators that act on the
bounce: dS6 of (Phi^+Phi)^3 and dSkin of (Phi^+Phi)(D_i Phi)^+(D_i Phi). With --add-dim6 the bounces of every
existing cache file without dSkin are solved again and the two shifts added; a file is refused (left unchanged,
reported) if a re-solved S3 differs from the cached one by more than 1e-6 relative, so that the actions themselves
cannot change."""
import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402  (src/, models/ and the folders of paper/reproduce on sys.path)
ROOT = paths.REPO

G = [0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.475, 0.48, 0.49, 0.5, 0.52, 0.55, 0.6, 0.65, 0.7, 0.8]


def scan_anchors():
    """{g: (v_line, anchor)} of the line-scan points at which TL percolated."""
    import glob as _glob
    import rate_history_u1 as R
    out = {}
    for d in _glob.glob(os.path.join(R.SCAN, "g*_u1_v*")):
        try:
            summ = json.load(open(os.path.join(d, "summary.json")))
        except FileNotFoundError:
            continue
        if summ.get("anchor_GeV") is not None:
            out[round(float(os.path.basename(d)[1:6]), 4)] = (float(summ["v_GeV"]), float(summ["anchor_GeV"]))
    return dict(sorted(out.items()))


def line_point(g):
    """(v_line, anchor) of a coupling: from its line-scan point if it has an anchor (TL's leading-order
    percolation temperature, or for g = 0.40 to 0.43 that of the direct integration); between
    two such points, v = e^{1/6} mu0/g and the anchor interpolated log-linearly in g; below the lowest,
    the anchor extrapolated log-linearly from the two lowest."""
    import rate_history_u1 as R
    try:
        summ = json.load(open(os.path.join(R.point_dir(g), "summary.json")))
        if summ.get("anchor_GeV") is not None:
            return float(summ["v_GeV"]), float(summ["anchor_GeV"])
    except FileNotFoundError:
        pass
    A = scan_anchors()
    lo = [x for x in A if x < g]
    hi = [x for x in A if x > g]
    if lo and hi:
        ga, gb = max(lo), min(hi)
        w = (g - ga) / (gb - ga)
        return (math.exp(1 / 6) * R.MU0 / g,
                math.exp((1 - w) * math.log(A[ga][1]) + w * math.log(A[gb][1])))
    # below the lowest coupling with an anchor: extrapolated log-linearly from the two lowest
    g0, g1 = sorted(A)[:2]
    return math.exp(1 / 6) * R.MU0 / g, A[g0][1] * math.exp((math.log(A[g1][1]) - math.log(A[g0][1])) / (g1 - g0) * (g - g0))
LOW_G = [0.39, 0.40, 0.41, 0.42, 0.43]
# 0.425 included: at small v it percolates above its _high caches
HIGHER_G = [0.39, 0.395, 0.40, 0.405, 0.41, 0.415, 0.42, 0.425]
HIGHEST_G = [0.39, 0.395, 0.40, 0.405]
# the couplings below the line's table between g = 0.38 and 0.405 (line_budget.G_EDGE, figure 1), on every grid
EDGE_G = [0.38, 0.385, 0.396, 0.3962, 0.3964, 0.3966, 0.3968, 0.397, 0.3972, 0.3975, 0.3978, 0.398, 0.399, 0.3995,
          0.3998, 0.4, 0.4001, 0.40015] + [round(0.4002 + 1e-5 * k, 5) for k in range(-4, 11)] \
         + [round(0.4003 + 1e-4 * k, 4) for k in range(1, 8)] + [0.402, 0.403, 0.404, 0.405]
KINDS = {"nlo1": (1.0, "NLO", True), "nlo05": (0.5, "NLO", False), "nlo2": (2.0, "NLO", False), "lom": (1.0, "LO", False),
         "mu3x05": (1.0, "NLO", False), "mu3x2": (1.0, "NLO", False)}
# the LO matching-scale band of the rate figure (g = 0.5, main grid; not used by the tables)
KINDS_FIG1 = {"lom05": (0.5, "LO", False), "lom2": (2.0, "LO", False)}
MU3_FACTOR = {"mu3x05": 0.5, "mu3x2": 2.0}      # the EFT scale mu3 = g T varied by a factor of two


#: the directory of the cache files (paper/data/action_cache, or ACTION_CACHE_DIR; build_exact_table.py reads the same)
CACHE_DIR = paths.ACTION_CACHE


def cache_tag(g):
    """The coupling in the name of a cache file: three decimals, or as many as the coupling has, up to six (g = 0.5625
    of the plane, g = 0.40015 of the edge of figure 1)."""
    for nd in (3, 4, 5, 6):
        if abs(round(g, nd) - g) < 1e-9:
            return f"{g:.{nd}f}"
    raise ValueError(f"g = {g} has more than six decimals")


def soft_params(g, kind, T):
    """The 3D parameters of `kind` at T: mu3 = g T with the input coupling g = g(mu0), as in the Delta S tables."""
    import rate_history_u1 as R
    mult, order, _ = {**KINDS, **KINDS_FIG1}[kind]
    return R.dr.matched_soft_parameters(g, 0.0, R.MU0, T, mu4_over_T=R.MU4 * mult, matching_order=order,
                                        mu3=g * T * MU3_FACTOR.get(kind, 1.0))


def lo_bounce(p, T):
    """The LO bounce as TransitionListener solves it (nlo_tables._bounce): in units of T, with the temporal coupling
    h3 = lambda_VL of the matching (TL's default temporal_coupling "h3"). Returns S3/T and r, phi3, info in GeV."""
    from transitionlistener.eft_rate import abelian_higgs_3d
    S3, r, phi3, info = abelian_higgs_3d.solve_LO_bounce(p["m3sq"] / T ** 2, p["lam3"] / T, p["g3sq"] / T,
                                              m_debye_sq=p["mD2"] / T ** 2, h3=p["lambda_vl"] / T)
    info = dict(info, dphi3=np.asarray(info["dphi3"]) * T ** 1.5)
    return S3, r / T, phi3 * math.sqrt(T), info


def dim6_shifts(p, r, phi3, info):
    """dS6 and dSkin on a LO bounce (r, phi3, info of abelian_higgs_3d.solve_LO_bounce) with the coefficients of p."""
    from transitionlistener.eft_rate import abelian_higgs_3d
    return dict(dS6=float(abelian_higgs_3d.dim6_action_shift(r, phi3, p["c6"])),
                dSkin=float(abelian_higgs_3d.dim6_kinetic_action_shift(r, phi3, info["dphi3"], p["c_D2phi4"])))


def add_dim6(path, rtol=1e-6):
    """Add dS6 and dSkin to an existing cache file (see the module docstring); returns a one-line report."""
    from transitionlistener.eft_rate import abelian_higgs_3d
    d = json.load(open(path))
    rows = d["rows"]
    if all("dSkin" in r for r in rows if np.isfinite(r["S3"])):
        return f"{os.path.basename(path)}: has dSkin"
    kind = d["kind"]
    worst, failed = 0.0, []
    for r in rows:
        if not np.isfinite(r["S3"]):
            continue
        p = soft_params(d["g"], kind, r["T"])
        try:
            S3, rr, phi3, info = lo_bounce(p, r["T"])
        except (ValueError, RuntimeError):
            # the cached bounce exists but the solver now finds no barrier (the marginal edge where the barrier
            # disappears): no shift, flagged and reported
            r.update(dS6=0.0, dSkin=0.0, dim6_resolve_failed=True)
            failed.append(f"T = {r['T']:.4e} GeV (cached S3 = {r['S3']:.4g})")
            continue
        worst = max(worst, abs(S3 / r["S3"] - 1.0))
        if worst > rtol:
            return f"{os.path.basename(path)}: REFUSED, S3 at T = {r['T']:.4e} GeV differs by {abs(S3 / r['S3'] - 1):.1e}"
        r.update(dim6_shifts(p, rr, phi3, info))
    tmp = path + f".tmp{os.getpid()}"
    json.dump(d, open(tmp, "w"))
    os.replace(tmp, path)
    msg = f"{os.path.basename(path)}: added, S3 reproduced to {worst:.1e}"
    return msg + (f"; NO BOUNCE ON RE-SOLVING at {len(failed)} temperatures: " + ", ".join(failed) if failed else "")


def task(args):
    g, kind = args[:2]
    low = len(args) > 2 and args[2]
    high = low == "high"
    higher = low == "high2"
    highest = low == "high3"
    from transitionlistener.eft_rate import abelian_higgs_3d
    import rate_history_u1 as R
    mult, order, prof = {**KINDS, **KINDS_FIG1}[kind]
    suffix = "_high3" if highest else "_high2" if higher else ("_high" if high else ("_low" if low else ""))
    out_path = os.path.join(CACHE_DIR, f"act_{kind}_g{cache_tag(g)}{suffix}.json")
    if os.path.exists(out_path):
        return out_path
    v_line, anchor = line_point(g)
    if highest:
        Ts = np.geomspace(anchor * 2000.0, anchor * 2e5, 41)[1:]
    elif higher:
        Ts = np.geomspace(anchor * 60.0, anchor * 2000.0, 41)[1:]
    elif high:
        Ts = np.geomspace(anchor * 8.0, anchor * 60.0, 41)[1:]
    elif low:
        Ts = np.geomspace(anchor * 1e-5, anchor * 0.05, 200, endpoint=False)
    else:
        Ts = np.geomspace(anchor * 0.05, anchor * 8.0, 140)
    rows = []
    for T in Ts:
        p = soft_params(g, kind, T)
        rec = {"T": float(T)}
        try:
            S3, r, phi3, info = lo_bounce(p, T)
            rec["S3"] = float(S3)
            rec.update(dim6_shifts(p, r, phi3, info))
            if prof:
                rec["x3"] = float(p["g3sq"] / (math.pi * math.sqrt(p["m3sq"]))) if p["m3sq"] > 0 else float("nan")
        except (ValueError, RuntimeError):
            rec["S3"] = float("nan")
        rows.append(rec)
    json.dump(dict(g=g, kind=kind, anchor=anchor, rows=rows), open(out_path, "w"))
    return out_path


if __name__ == "__main__":
    os.makedirs(os.path.join(CACHE_DIR), exist_ok=True)
    if "--add-dim6" in sys.argv:            # the two dimension-six shifts for the existing caches
        import glob
        files = sorted(f for f in glob.glob(os.path.join(CACHE_DIR, "act_*_g*.json"))
                       if os.path.basename(f).split("_g")[0][4:] in {**KINDS, **KINDS_FIG1})
        with Pool(min(len(files), int(os.environ.get("ACTION_WORKERS", "60")))) as pool:
            for line in pool.imap_unordered(add_dim6, files):
                print(line, flush=True)
        sys.exit(0)
    if "--edge" in sys.argv:             # the edge couplings of figure 1, every grid
        jobs = [(g, k) + ((s,) if s is not None else ()) for g in EDGE_G for k in KINDS
                for s in (None, True, "high", "high2", "high3")]
    elif "--fig1" in sys.argv:           # the LO matching-scale band of the rate figure at g = 0.5 (main grid)
        jobs = [(0.5, k) for k in KINDS_FIG1]
    elif "--highest" in sys.argv:          # the faster variants at the smallest couplings
        jobs = [(g, k, "high3") for g in HIGHEST_G for k in KINDS]
    elif "--higher" in sys.argv:           # the further extension for the slow couplings at small v
        jobs = [(g, k, "high2") for g in HIGHER_G for k in KINDS]
    elif "--high" in sys.argv:             # the extension above the grid for every coupling of the table
        import build_exact_table as _BET
        jobs = [(g, k, "high") for g in _BET.G for k in KINDS]
    elif "--g" in sys.argv:                # given couplings, main grid and low-temperature extension
        gs = [float(x) for x in sys.argv[sys.argv.index("--g") + 1].split(",")]
        jobs = [(g, k) for g in gs for k in KINDS] + [(g, k, True) for g in gs for k in KINDS]
    elif "--low-all" in sys.argv:          # the extension for every coupling (latc-, latp-, fulllo at g = 0.44)
        jobs = [(g, k, True) for g in G for k in KINDS]
    elif "--low" in sys.argv:
        jobs = [(g, k, True) for g in LOW_G for k in KINDS]
    else:
        jobs = [(g, k) for g in G for k in KINDS]
    jobs = sorted(jobs, key=lambda j: not os.path.exists(os.path.join(CACHE_DIR, f"act_{j[1]}_g{cache_tag(j[0])}"
                                                                   f"{({'high': '_high', 'high2': '_high2', 'high3': '_high3'}.get(j[2], '_low')) if len(j) > 2 else ''}.json")))
    with Pool(min(len(jobs), 60)) as pool:
        for p in pool.imap_unordered(task, jobs):
            print("done", os.path.basename(p), flush=True)
