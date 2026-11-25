"""Transition parameters of the four-dimensional daisy-resummed rate (variant N5 of the paper; N4 with
--kind bounce_rg) on the (g, log10 v) plane, by exact percolation, in the layout of the exact_table.npz of
rate/build_exact_table.py.

Rate: Gamma = T^4 (S3/(2 pi T))^{3/2} e^{-S3/T}, TransitionListener's 4D rate
(bubbledynamics.logGamma, the default rate_mode "S3overT_4D"), with S3/T from
compute_daisy_actions.py. At fixed g, S3/T depends on T/v only; v enters through H_V ~ v^2/M_Pl and
the Standard Model degrees of freedom at the real temperature, exactly as for the NLO table.

Percolation: rate/exact_percolation.percolate (the false-vacuum fraction
on the self-consistent mean background, clock ln a, radiation with the SM table plus four dark
degrees of freedom, a ~ 1/(T g_s^{1/3})), and the per-node outputs of
build_exact_table.run_g, both imported unchanged: run_g is called with build_exact_table.Model
replaced by DaisyModel below and build_exact_table.VARIANTS by ["daisy"], so the keys, the
window_limited / T_min_window / f_window_limited logic and T_reh_SM, T_reh_SM4 are those of the NLO
table. in_table is 1 everywhere (there is no Delta S table).

Window: the temperatures at which the action is known, as for the NLO table. For the tl action
(TransitionListener's solver) that ends at the lowest temperature with a finite action,
T/v = 1.3e-4 at g = 0.5 to 2.1e-4 at g = 0.8 (compute_daisy_actions.py: below it TL's tunnelling
returns an infinite action, a solver limitation), so that every node of daisy_table.npz without
percolation is window_limited: the rate still rises where TL's action ends. A node whose transition does not percolate before that temperature, or
percolates within a factor 1.5 of it, is window_limited = 1 (inconclusive), as in the NLO table.

Usage (tables written to the work folder; prepare_daisy_table.py turns them into paper/data/daisy_*plane*.npz):
  python build_daisy_table.py                      daisy_table.npz from <DAISY_CACHE>/daisy_tl_g*.json
  python build_daisy_table.py --kind direct        daisy_direct_table.npz (DIAGNOSTIC action, window
                                                   T/v >= 1e-8; VARIANTS = ["daisy_direct"])
  python build_daisy_table.py --kind bounce        daisy_bounce_table.npz: the bounce action for
                                                   T_int <= 1 joined to the direct one above, window
                                                   T/v >= 1e-13; VARIANTS = ["daisy_bounce"]
  python build_daisy_table.py --kind bounce_rg     daisy_bounce_rg_table.npz: the bounce action of the potential with
                                                   the couplings run to mu = pi T (models/TL_conformal_u1_no_fermions_rg.py)
                                                   on its whole grid, no join; VARIANTS = ["daisy_bounce_rg"]
  python build_daisy_table.py --probe              edge probe on <DAISY_CACHE>/daisy_tlprobe_g*.json,
                                                   written to <work>/edge_probe.txt
"""
import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.dont_write_bytecode = True     # no __pycache__ in the directories this script imports from
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402  (src/, models/ and the folders of paper/reproduce on sys.path)
import exact_percolation as EP  # noqa: E402
import build_exact_table as BET  # noqa: E402
import compute_daisy_actions as CDA  # noqa: E402

LOGV = BET.LOGV
T_INT_CONTINUE = 1e-4      # "continued" window: zero rate below the solver cutoff, down to this T_int


class DaisyModel:
    """The 4D daisy action of one coupling, with the attributes build_exact_table.run_g reads from
    its Model: T (GeV at v_line), exponent(variant), v_line, dV_line, rm, table_T, T_scale.

    window = "solver"     the temperatures where the action is finite and above (above T_c the
                          action is infinite too; the rate there is zero, which is harmless)
             "continued"  the whole grid with the infinite actions kept as zero rate, extended
                          down to T_int = T_INT_CONTINUE: what TransitionListener's percolation
                          does with the tl action (the bubbles nucleated above the cutoff keep
                          growing below it). Used for the comparison with the TL line runs."""

    def __init__(self, g, kind="tl", tag="", window="solver"):
        d = json.load(open(CDA.cache_path(g, kind, tag)))
        if kind == "bounce":
            # the bounce action up to its top (T_int = 1), the direct action above, where its
            # tolerance bias is below 3e-4 in S3/T
            top = max(r["T_int"] for r in d["rows"])
            hot = [r for r in json.load(open(CDA.cache_path(g, "direct")))["rows"] if r["T_int"] > top * 1.0001]
            d["rows"] = d["rows"] + hot
        rows = sorted(d["rows"], key=lambda r: r["T_int"])
        T_int = np.array([r["T_int"] for r in rows], float)
        S = np.array([r["S3_over_T"] if r["S3_over_T"] is not None else np.nan for r in rows], float)
        if any(r["trantype"] == 2 for r in rows):
            raise RuntimeError(f"g={g}: a temperature without a barrier (zero action) in the grid")
        fin = np.isfinite(S)
        if window == "solver":
            keep = T_int >= T_int[fin].min()
        elif window == "continued":
            keep = np.ones_like(fin)
            lo = np.geomspace(T_INT_CONTINUE, T_int[0], 60, endpoint=False)
            T_int = np.concatenate([lo, T_int]); S = np.concatenate([np.full(len(lo), np.nan), S])
            keep = np.concatenate([np.ones(len(lo), bool), keep])
        else:
            raise ValueError(window)
        cf = d["conversionFactor"]
        self.g = g
        self.T = T_int[keep] * cf                # GeV at v_line
        self._S = S[keep]
        self.v_line = d["v_ref_GeV"]
        self.dV_line = d["dV_GeV4"]
        self.T_int_cut = float(T_int[np.isfinite(S)].min())
        self.rm = EP.RateModel.__new__(EP.RateModel)
        self.table_T = (0.0, np.inf)             # in_table = 1: no Delta S table to leave
        self.T_scale = 1.0

    def exponent(self, variant):
        return self._S

    def in_range(self, T_line):
        """in_table = 1 everywhere: there is no Delta S table to leave (build_exact_table.run_g asks)."""
        return True


def build(gs, kind, tag, out_path, workers):
    """build_exact_table.run_g on DaisyModel for every coupling; the npz layout of exact_table.npz."""
    BET.VARIANTS = ["daisy" if kind == "tl" else f"daisy_{kind}"]
    BET.Model = lambda g: DaisyModel(g, kind, tag)
    t0 = time.time()
    with Pool(min(workers, len(gs))) as pool:
        res = pool.map(BET.run_g, gs)
    res.sort(key=lambda x: x[0])
    # copied from build_exact_table.py's __main__ (the npz writer), with G and VARIANTS of this table
    tab = {key: np.array([r[1][key] for r in res]) for key in BET.KEYS}      # (g, variant, logv)
    np.savez(out_path, G=np.array(gs), LOGV=LOGV, VARIANTS=np.array(BET.VARIANTS),
             T_reh_SM=np.array([r[2][0] for r in res]), T_reh_SM4=np.array([r[2][1] for r in res]), **tab)
    j = int(np.argmin(np.abs(LOGV - 0.25)))
    for (g, o, tr) in res:
        print(f"g={g:.3f} v=10^{LOGV[j]}: T_p {o['T_p'][0, j]:.3e} (b/H)_R {o['betaH_R'][0, j]:.2f} "
              f"(b/H)_S {o['betaH_S'][0, j]:.2f} wl {o['window_limited'][0, j]:.0f}; (b/H)_R vs v: "
              + " ".join(f"{x:.1f}" for x in o["betaH_R"][0]), flush=True)
    print(f"wrote {out_path}, {time.time() - t0:.0f} s", flush=True)


def percolate_at(g, v, kind="tl", tag="", window="solver"):
    """exact_percolation.percolate of the daisy rate at (g, v) in GeV, with run_g's T_p < 1.5 x
    window rule reported separately (key 'near_window')."""
    m = DaisyModel(g, kind, tag, window)
    rm = m.rm
    rm.g, rm.v_line, rm.dV_line, rm.Ts, rm.corr = g, m.v_line, m.dV_line, m.T, None
    S = m.exponent("daisy")
    rm.S3 = S
    rm.exponent = lambda T_line, variant, S=S: np.interp(np.log(T_line), np.log(m.T), S, left=np.nan, right=np.nan)
    r = EP.percolate(rm, v=v)
    r.pop("tau_grid", None); r.pop("lnG_grid", None)
    r["T_cut_GeV"] = m.T_int_cut * v / 1000.0          # the lowest temperature with a finite action, at this v
    r["near_window"] = bool(r.get("ok") and r["T_p"] < 1.5 * r["T_cut_GeV"])
    return r


def _probe_one(args):
    g, lv, window = args
    try:
        return g, lv, window, percolate_at(g, 10.0 ** lv, "tl", "probe", window)
    except FileNotFoundError:
        return g, lv, window, None


def probe(workers):
    """Where the tl rate stops percolating at the two ends of the v range and on the line."""
    import glob
    gs = sorted(float(os.path.basename(p)[len("daisy_tlprobe_g"):-5]) for p in glob.glob(os.path.join(CDA.CACHE, "daisy_tlprobe_g*.json")))
    jobs = [(g, lv, w) for g in gs for lv in (LOGV[0], round(math.log10(CDA.v_ref(g)), 4), LOGV[-1]) for w in ("solver", "continued")]
    with Pool(min(workers, len(jobs))) as pool:
        res = pool.map(_probe_one, jobs)
    lines = ["edge probe of the tl action (20 temperatures per decade): T_p in GeV, P_true at the end of the window"]
    for g, lv, w, r in res:
        if r is None:
            continue
        lines.append(f"g={g:.3f} log10v={lv:+.3f} {w:9s}: ok={r['ok']} T_p={r.get('T_p', float('nan')):.3e} "
                     f"T_cut={r.get('T_cut_GeV', float('nan')):.3e} near_window={r.get('near_window')} "
                     f"P_t_end={r.get('P_t_end', float('nan')):.3f} T_f={r.get('T_f', float('nan')):.3e} "
                     f"bH_R={r.get('betaH_R', float('nan')):.2f}")
    open(paths.work("edge_probe.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=("tl", "direct", "bounce", "bounce_rg"), default="tl")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--workers", type=int, default=56)
    a = ap.parse_args()
    if a.probe:
        probe(a.workers)
    else:
        gs = CDA.G_D if a.kind == "tl" else CDA.G_RG if a.kind == "bounce_rg" else CDA.G_DIRECT
        out = paths.work({"tl": "daisy_table.npz", "direct": "daisy_direct_table.npz",
                                  "bounce": "daisy_bounce_table.npz", "bounce_rg": "daisy_bounce_rg_table.npz"}[a.kind])
        build(gs, a.kind, "", out, a.workers)
