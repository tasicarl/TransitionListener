"""Numbers of sections 2 and 3.1 read directly from the line results: the NLO transition at the smallest coupling of
the line, the couplings below which the NLO and the two 4D daisy rates (N4: couplings run to pi T, N5: fixed
couplings) no longer complete on the line, N4 and N5 at g = 0.5 against NLO, and the shift of T_p under the
prefactor variation (A_dyn x 10 and / 10) along the line.
Inputs: data/line_budget.json (rate/line_budget.py), data/daisy_line.json and data/daisy_line_rg.json
(daisy/daisy_line.py).
    python line_daisy_numbers.py"""
import json
import os

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
L = json.load(open(os.path.join(DATA, "line_budget.json")))
N5 = json.load(open(os.path.join(DATA, "daisy_line.json")))
N4 = json.load(open(os.path.join(DATA, "daisy_line_rg.json")))
byg = {round(o["g"], 4): o for o in L}

o = byg[0.40]["nlo"]
print(f"NLO at g = 0.40: T_p {o['T_p'] * 1e12:.2f} meV, (beta/H)_S {o['betaH_S']:.3f}, (beta/H)_R {o['betaH_R']:.3f}, "
      f"(beta/H)_Gamma {o['beta_Gamma_H']:.3f}")

g_nlo = min(round(x["g"], 4) for x in L if x["nlo"] and x["nlo"].get("T_f") is not None)
for name, rows in (("N5 (4D daisy, fixed couplings)", N5), ("N4 (4D daisy, couplings at pi T)", N4)):
    done = sorted(r["g"] for r in rows if r["ok"] and r.get("T_f") is not None)
    print(f"{name}: smallest coupling on the line that completes {done[0]:.4f} "
          f"({100 * (done[0] / g_nlo - 1):+.0f}% against NLO, {g_nlo:.4f})")

nlo = byg[0.5]["nlo"]
for name, rows in (("N4", N4), ("N5", N5)):
    r = next(r for r in rows if abs(r["g"] - 0.5) < 1e-9)
    print(f"{name} at g = 0.5: T_p {r['T_p'] * 1e6:.2f} keV, (beta/H)_R {r['betaH_R']:.2f}; NLO T_p "
          f"{nlo['T_p'] * 1e6:.2f} keV, (beta/H)_R {nlo['betaH_R']:.2f}; ratio of the (beta/H)_R {r['betaH_R'] / nlo['betaH_R']:.2f}")

print("T_p shift under A_dyn x 10 and / 10 (half the difference, per cent), along the tabulated couplings:")
for g in sorted(byg):
    x = byg[g]
    if x["nlo"] and not x.get("figure_only") and 0.415 < g < 0.505:
        print(f"  g = {g:.3f}: {x['entries_pct']['T_p']['pref']:.1f}%")
