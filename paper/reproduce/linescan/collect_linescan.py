"""Collect the transition parameters of a line scan into the JSON that
plane_scan.py and figs/make_figures.py read.

  python collect_linescan.py <points root> [<points root> ...] --out <json> [--v 0.173] [--tag y0.01] [--any-v]

The default tag and v select the scan of the model with fermions (scan_point.py
--model dark_u1, y = 0.01, v = 0.173 GeV). For the fermion-free model of the
paper (scan_point.py --model u1, v = e^{1/6} mu0 / g) use

  python collect_linescan.py <paths.LINESCAN_POINTS> --tag u1 --any-v \\
         --out ../../data/linescan_u1_mu0_1GeV.json

and pass the JSON to plane_scan.py with --linescan. Only points whose
next-to-leading order percolation run finished are kept; the rest are listed.
"""
import argparse
import glob
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("roots", nargs="+")
ap.add_argument("--out", required=True)
ap.add_argument("--v", type=float, default=0.173, help="GeV; ignored with --any-v")
ap.add_argument("--tag", default="y0.01")
ap.add_argument("--any-v", action="store_true", help="accept any v (the campaign has v = e^{1/6}/g)")
args = ap.parse_args()

rows, skipped = {}, []
for root in args.roots:
    pattern = f"g*_{args.tag}_v*/summary.json" if args.any_v else f"g*_{args.tag}_v{args.v:g}/summary.json"
    for f in sorted(glob.glob(os.path.join(root, pattern))):
        s = json.load(open(f))
        g = s["g"]
        if g in rows:
            continue

        def obs(mode, mu):
            return (s.get(mode, {}).get(mu, {}) or {}).get("observables") or {}

        c = obs("nlo3d", "1")
        if not c.get("Tperc_SM_GeV"):
            skipped.append((g, (s.get("nlo3d", {}).get("1", {}) or {}).get("error") or s.get("stopped", "no nlo3d")))
            continue
        r = dict(g=g, v_GeV=s["v_GeV"], model=s.get("model", "dark_u1"),
                 Tperc=c["Tperc_SM_GeV"], Treh=c["Treh_SM_GeV"], alpha=c["alpha"],
                 betaH=c["betaH_S3"], betaH_RH=c.get("betaH_RH"),
                 frac=(s.get("uncertainty") or {}).get("Tperc_fraction"),
                 src=os.path.relpath(f, os.path.dirname(os.path.abspath(args.out))))
        for mode, mu, key in (("nlo3d", "0.5", "mu05"), ("nlo3d", "2", "mu2"), ("lo3d", "1", "lo"), ("4d", "1", "4d")):
            o = obs(mode, mu)
            r[f"Tperc_{key}"] = o.get("Tperc_SM_GeV"); r[f"betaH_{key}"] = o.get("betaH_S3")
            r[f"Treh_{key}"] = o.get("Treh_SM_GeV"); r[f"alpha_{key}"] = o.get("alpha")
        r["S3_LO_at_Tp"] = (s.get("uncertainty") or {}).get("S3_LO")
        r["delta_S"] = (s.get("uncertainty") or {}).get("delta_S")
        rows[g] = r
json.dump({str(g): rows[g] for g in sorted(rows)}, open(args.out, "w"), indent=1)
print(f"{len(rows)} couplings written to {args.out}: {sorted(rows)}")
for g, why in skipped:
    print(f"  skipped g = {g}: {str(why)[:100]}")
