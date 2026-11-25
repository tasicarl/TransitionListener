"""The numbers of the text that come from the line v = e^{1/6} GeV/g (data/line_budget.json, direct integration, with
TransitionListener's line runs in the entry TL). Sections 2.2, 2.3 and 3.1, appendix A."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
L = [o for o in json.load(open(os.path.join(HERE, "..", "..", "data", "line_budget.json"))) if o["nlo"]]
T = {round(o["g"], 4): o for o in L if not o.get("figure_only")}


def rng(gmin, gmax, f):
    x = np.array([f(o) for o in L if gmin - 1e-9 <= o["g"] <= gmax + 1e-9], float)
    x = x[np.isfinite(x)]
    return x.min(), x.max()


print("|delta S_tot| g 0.44-0.5: %.2f-%.2f" % rng(.44, .5, lambda o: abs(o["S_nlo_minus_lo_at_Tp"])))
print("Delta S g 0.44-0.5: %.2f-%.2f" % rng(.44, .5, lambda o: o["delta_S_at_Tp"]))
print("x3 at T_p g 0.45-0.5: %.3f-%.3f" % rng(.45, .5, lambda o: o["x3_at_Tp"]))
print("x3 |delta S| g 0.44-0.5: %.2f-%.2f" % rng(.44, .5, lambda o: o["x3_at_Tp"] * abs(o["S_nlo_minus_lo_at_Tp"])))
print("exponent S_p g 0.42-0.8: %.2f-%.2f" % rng(.42, .8, lambda o: o["nlo"]["S_p"]))
o = T[0.45]["nlo"]
print("g 0.45: (b/H)_G %.3f (b/H)_R %.3f (b/H)_S %.3f, S-G %.2f, S-R %.2f, T_p %.4g keV" % (o["beta_Gamma_H"], o["betaH_R"], o["betaH_S"], o["betaH_S"] - o["beta_Gamma_H"], o["betaH_S"] - o["betaH_R"], o["T_p"] * 1e6))
print("T_p at g 0.5: %.4g GeV" % T[0.5]["nlo"]["T_p"])


def pct(q, var):
    return lambda o: 100 * (o[var][q] / o["nlo"][q] - 1) if o.get(var) else np.nan


for name, e in (("mu4", "mu4"), ("mu3", "mu3"), ("dim6", "dim6_signed"), ("hord", "hord"), ("pref", "pref"), ("lo", "lo_signed")):
    for q in ("T_p", "betaH_R"):
        a, b = rng(.44, .5, lambda o: abs(o["entries_pct"][q][e]))
        print(f"|entry {name} {q}| g 0.44-0.5: {a:.2f}-{b:.2f}")
print("|LO change of (b/H)_R| g 0.42-0.8: %.2f-%.2f" % rng(.42, .8, lambda o: abs(o["entries_pct"]["betaH_R"]["lo_signed"])))
print("exponent shift of prefactor x10: %.3f" % np.log(10))
print("LO -> NLO: Delta (b/H)_R / (b/H)_R over relative shift of exponent g 0.44-0.5")
print("  ratio: %.2f-%.2f" % rng(.44, .5, lambda o: (o["entries_pct"]["betaH_R"]["lo_signed"] / 100) / (o["S_nlo_minus_lo_at_Tp"] / o["nlo"]["S_p"]) * -1 if o["entries_pct"]["betaH_R"]["lo_signed"] else np.nan))
d = [(o["g"], o["entries_pct"]["betaH_R"]["lo_signed"]) for o in L if .44 <= o["g"] <= .5]
print("  LO change of (b/H)_R by g:", [(g, round(x, 2)) for g, x in d])
print("TL g 0.45-0.8: max |dT_p| %.2f %%, max |d(b/H)_R| %.2f %%" % (
    max(abs(100 * (o["TL"]["T_p"] / o["nlo"]["T_p"] - 1)) for k, o in T.items() if .45 <= k <= .8 and "TL" in o),
    max(abs(100 * (o["TL"]["betaH_R"] / o["nlo"]["betaH_R"] - 1)) for k, o in T.items() if .45 <= k <= .8 and "TL" in o)))
print("TL g 0.42-0.44: max |dT_p| %.2f %%, max |d(b/H)_R| %.2f %%" % (
    max(abs(100 * (o["TL"]["T_p"] / o["nlo"]["T_p"] - 1)) for k, o in T.items() if .42 <= k <= .44 and "TL" in o),
    max(abs(100 * (o["TL"]["betaH_R"] / o["nlo"]["betaH_R"] - 1)) for k, o in T.items() if .42 <= k <= .44 and "TL" in o)))
