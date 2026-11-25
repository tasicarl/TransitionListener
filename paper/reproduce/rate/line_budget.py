"""The theory budget along the line scan (v = e^{1/6} mu0/g, mu0 = 1 GeV) by exact percolation.

For every coupling of G and every rate variant of build_exact_table.py in VARS, the false-vacuum
fraction is integrated at the line v itself (no interpolation in v). Reports T_p, the three
measures of beta/H, the completion measure, Delta S at T_p, the LO -> NLO shifts (the LO bounce
on the NLO-matched theory, and the full LO), and the theory entries in per cent of T_p and of
(beta/H)_R: half the mu4 and mu3 bands, the next order (hord: x3 |S_NLO - S_LO|, and next:
x3 max(|Delta S|, 1), each as the larger of its two excursions), dimension six (the default against
dim4) and the prefactor (a factor of ten either way, half the difference). The total band is the
quadrature sum of mu4, mu3, dimension six and hord.
For the validation each coupling also carries TransitionListener's own T_p and (beta/H)_R: "TL" from
<work>/accuracy_table<TL_ACC_TAG>.json (linescan/accuracy_table.py, TransitionListener with its defaults;
no tag by default), "TL_before" from the line scan of linescan/scan_point.py
(paper/data/linescan_u1_mu0_1GeV.json).
Writes paper/data/line_budget.json. With --tl-only, refreshes only the TL entries of an existing line_budget.json."""
import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402  (src/, models/ and the folders of paper/reproduce on sys.path)
import build_exact_table as B  # noqa: E402

G = [0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.475, 0.48, 0.49, 0.5, 0.52, 0.55, 0.6, 0.65, 0.7, 0.8]
# couplings for the figure only (not in the tables): the remaining couplings of the plane table and six more, so
# that the curves resolve the e+- annihilation (T_p = 50 to 600 keV, g = 0.48 to 0.52) and the muon annihilation
# (g = 0.6 to 0.75) instead of drawing a straight line across them
G_FIG = [0.485, 0.495, 0.505, 0.51, 0.515, 0.535, 0.5625, 0.575, 0.5875, 0.625, 0.675, 0.725, 0.75, 0.775]
# and every coupling on a grid of 0.005, for smooth curves
G_FIG = sorted(set(G_FIG) | {round(float(x), 4) for x in np.arange(0.42, 0.8001, 0.005)} - set(G))
# and below the line's smallest table coupling, down to where the NLO rate no longer completes (g = 0.3964), densely
# where the curves of figure 1 bend
# (compute_actions.py --edge computes their actions)
from compute_actions import EDGE_G  # noqa: E402
G_EDGE = sorted(set(EDGE_G) | {0.39, 0.395, 0.41, 0.415})
G_FIG = sorted(set(G_FIG) | set(G_EDGE))
G = sorted(set(G) | set(G_FIG))
VARS = ["nlo", "lorate", "fulllo", "mu05", "mu2", "mu3x05", "mu3x2", "next+", "next-", "hord+", "hord-", "dim4", "pref+", "pref-", "adynT"]
LS = json.load(open(paths.data("linescan_u1_mu0_1GeV.json")))
# TL_ACC_TAG selects the file of accuracy_table.py that holds the TL runs (default: no tag)
_ACC = paths.work(f"accuracy_table{os.environ.get('TL_ACC_TAG', '')}.json")
TLF = {round(r["g"], 3): r for r in json.load(open(_ACC)) if "T_p_TL" in r} if os.path.exists(_ACC) else {}


def run(g):
    m = B.Model(g)
    rm = m.rm
    rm.g, rm.v_line, rm.dV_line, rm.Ts, rm.corr = g, m.v_line, m.dV_line, m.T, None
    out = {"g": g, "v_GeV": m.v_line, "figure_only": g in G_FIG}
    for var in VARS:
        S = m.exponent(var)
        B.sample(rm, m.T, S)
        r = B.EP.percolate(rm, v=m.v_line)
        if not r["ok"]:
            out[var] = None
            continue
        out[var] = {k: float(r[k]) for k in ("T_p", "T_nuc", "T_f", "betaH_S", "betaH_R", "beta_Gamma_H", "dlnV_dt_over_H", "alpha", "S_p")}
    S3 = m.S3["nlo1"]
    dS = m.dS("1", S3)
    if out["nlo"]:
        Tp = out["nlo"]["T_p"]
        out["delta_S_at_Tp"] = float(np.interp(math.log(Tp), np.log(m.T), dS))
        # the whole step from LO to NLO in the exponent at the NLO percolation temperature:
        # NLO matching of the bounce plus Delta S, against the LO bounce on the LO-matched theory
        s_nlo, s_lo = m.exponent("nlo"), m.exponent("fulllo")
        out["S_nlo_minus_lo_at_Tp"] = float(np.interp(math.log(Tp), np.log(m.T), s_nlo - s_lo))
        out["x3_at_Tp"] = float(np.interp(math.log(Tp), np.log(m.T), m.x3))
        out["in_table"] = m.in_range(Tp)
    key = f"{g:g}"
    if key in LS and LS[key].get("Tperc"):
        out["TL_before"] = {"T_p": LS[key]["Tperc"], "betaH_R": LS[key].get("betaH_RH"), "betaH_S": LS[key].get("betaH")}
    if round(g, 3) in TLF:
        out["TL"] = {"T_p": TLF[round(g, 3)]["T_p_TL"], "betaH_R": TLF[round(g, 3)]["betaH_RH_TL"]}
    return out


def pct(a, b):
    return 100.0 * (a / b - 1.0)


if __name__ == "__main__" and "--tl-only" in sys.argv:
    res = json.load(open(paths.data("line_budget.json")))
    for o in res:
        o.pop("TL", None)
        if round(o["g"], 3) in TLF:
            o["TL"] = {"T_p": TLF[round(o["g"], 3)]["T_p_TL"], "betaH_R": TLF[round(o["g"], 3)]["betaH_RH_TL"]}
    json.dump(res, open(paths.data("line_budget.json"), "w"), indent=1)
    print("TL entries refreshed from", os.path.basename(_ACC), "at", sorted(TLF))
elif __name__ == "__main__":
    with Pool(min(len(G), int(os.environ.get("WORKERS", "28")))) as pool:
        res = pool.map(run, G)
    for o in res:
        n = o["nlo"]
        if not n:
            print(f"g={o['g']}: no percolation at the line"); continue
        e = {}
        for q in ("T_p", "betaH_R"):
            get = lambda v: o[v][q] if o.get(v) else np.nan
            e[q] = dict(
                lo_bounce=pct(n[q], get("lorate")), full_lo=pct(n[q], get("fulllo")),
                mu4=0.5 * abs(pct(get("mu2"), n[q]) - pct(get("mu05"), n[q])),
                mu3=0.5 * abs(pct(get("mu3x2"), n[q]) - pct(get("mu3x05"), n[q])),
                next=max(abs(pct(get("next+"), n[q])), abs(pct(get("next-"), n[q]))),
                # the larger of the two excursions: near the e+- kink both signs move the speed the same way,
                # and half their difference would nearly vanish there
                hord=max(abs(pct(get("hord+"), n[q])), abs(pct(get("hord-"), n[q]))),
                dim6=abs(pct(get("dim4"), n[q])),     # default (with dimension six) against dim4
                pref=0.5 * abs(pct(get("pref+"), n[q]) - pct(get("pref-"), n[q])),
                adynT=pct(get("adynT"), n[q]),
                # signed changes of the NLO value under one alternative, X_alt/X_NLO - 1 (table 1)
                lo_signed=pct(get("fulllo"), n[q]), dim6_signed=-pct(get("dim4"), n[q]),
                pref10_signed=pct(get("pref+"), n[q]))
            e[q]["total"] = math.sqrt(sum(e[q][k] ** 2 for k in ("mu4", "mu3", "dim6", "hord")))
        o["entries_pct"] = e
        tl = o.get("TL", {})
        print(f"g={o['g']:.3f} v={o['v_GeV']:.3f}: T_p {n['T_p']:.3e} (TL {tl.get('T_p', float('nan')):.3e}, {pct(n['T_p'], tl['T_p']) if tl else float('nan'):+.1f}%) "
              f"(b/H)_R {n['betaH_R']:.2f} (TL {tl.get('betaH_R') or float('nan'):.2f}) (b/H)_S {n['betaH_S']:.2f} bG {n['beta_Gamma_H']:.2f} "
              f"dlnV {n['dlnV_dt_over_H']:+.2f} dS {o.get('delta_S_at_Tp', float('nan')):+.2f} | T_p %: LO {e['T_p']['lo_bounce']:+.1f} fullLO {e['T_p']['full_lo']:+.1f} "
              f"mu4 {e['T_p']['mu4']:.1f} mu3 {e['T_p']['mu3']:.1f} next {e['T_p']['next']:.1f} dim6 {e['T_p']['dim6']:.1f} tot {e['T_p']['total']:.1f} pref {e['T_p']['pref']:.1f} "
              f"| b/H %: LO {e['betaH_R']['lo_bounce']:+.1f} fullLO {e['betaH_R']['full_lo']:+.1f} tot {e['betaH_R']['total']:.2f} pref {e['betaH_R']['pref']:.1f}", flush=True)
    json.dump(res, open(paths.data("line_budget.json"), "w"), indent=1)
