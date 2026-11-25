"""Accuracy of TransitionListener against the direct integration along the paper's line.

TL: <work>/tl_capture/capture_g<g><tag>.npz (capture_tl.py, TransitionListener with its defaults). Reference:
exact_percolation.percolate with the rate at ln a of the background and the plasma's radiation
(clock="lna", radiation=True), same rate, same v. Writes <work>/accuracy_table<tag>.json, which
rate/line_budget.py reads, and prints the table.
Usage: python accuracy_table.py [tag]  (default: no tag)"""
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402  (src/, models/ and the folders of paper/reproduce on sys.path)
import exact_percolation as EP  # noqa: E402

tag = sys.argv[1] if len(sys.argv) > 1 else ""
rows = []
for f in sorted(glob.glob(paths.work("tl_capture", f"capture_g*{tag}.npz"))):
    m = re.search(r"capture_g([0-9.]+)" + re.escape(tag) + r"\.npz$", f)
    if not m:
        continue
    g = float(m.group(1))
    C = np.load(f)
    keys = list(C["obs_keys"]); vals = C["obs_vals"]
    obs = dict(zip(keys, vals))
    tl_Tp = float(C["Tp"]) * float(C["conv"])
    if not np.isfinite(float(obs.get("betaH_RH", np.nan))):
        # the run stopped after the mean bubble separation (the strength failed at this
        # temperature); (beta/H)_RH follows from the captured R_* with v_w = 1
        RH = float(C["R"]) * float(np.interp(C["Tp"], C["T"], C["H"]))
        obs["betaH_RH"] = (8 * np.pi / EP.F_PERC) ** (1 / 3) / RH
    r = EP.percolate(EP.RateModel(g), v=float(C["v"]), clock="lna", radiation=True)
    if not r.get("ok", True) or "T_p" not in r:
        rows.append(dict(g=g, note="no percolation in the reference window"))
        continue
    row = dict(g=g, v_GeV=float(C["v"]), T_p_TL=tl_Tp, T_p_exact=r["T_p"],
               dT_p_pct=100 * (tl_Tp / r["T_p"] - 1),
               betaH_RH_TL=float(obs.get("betaH_RH", np.nan)), betaH_RH_exact=r["betaH_R"],
               dbetaH_RH_pct=100 * (float(obs.get("betaH_RH", np.nan)) / r["betaH_R"] - 1),
               alpha_exact=r["alpha"])
    rows.append(row)
json.dump(rows, open(paths.work(f"accuracy_table{tag}.json"), "w"), indent=1)
print(f"{'g':>6} {'v/GeV':>8} {'T_p TL/GeV':>12} {'T_p exact':>12} {'dT_p %':>8} {'(b/H)_RH TL':>12} {'exact':>9} {'d %':>7} {'alpha':>9}")
for r in rows:
    if "note" in r:
        print(f"{r['g']:6.3f}  {r['note']}")
        continue
    print(f"{r['g']:6.3f} {r['v_GeV']:8.4f} {r['T_p_TL']:12.5e} {r['T_p_exact']:12.5e} {r['dT_p_pct']:+8.2f} "
          f"{r['betaH_RH_TL']:12.3f} {r['betaH_RH_exact']:9.3f} {r['dbetaH_RH_pct']:+7.2f} {r['alpha_exact']:9.3g}")
ok = [r for r in rows if "note" not in r and np.isfinite(r["dbetaH_RH_pct"])]
if ok:
    print(f"max |dT_p| = {max(abs(r['dT_p_pct']) for r in ok):.2f} %, max |d(beta/H)_RH| = {max(abs(r['dbetaH_RH_pct']) for r in ok):.2f} % over {len(ok)} couplings")
