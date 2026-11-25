"""Reference values of the individual channels of the assembled exponent
(appendix B, eq. assembly) at a few named points of the fermion-free model, so
that an independent implementation can be compared term by term. Recomputes
the channels as TransitionListener does with its defaults (nlo_tables.fast_row and mixing_row: h3 in the temporal
sector, g3^2 at NLO) at the percolation temperature of the NLO rate (the variant "nlo" of paper/data/line_budget.json,
from the direct integration), with Delta S of the grid the model uses. Writes
output/determinant_reference_values.json and paper/data/determinant_reference_values.csv (table tab:reference)."""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402  (src/ and models/ on sys.path)
from transitionlistener.eft_rate import abelian_higgs_3d  # noqa: E402
from transitionlistener.eft_rate import nlo_tables as nt  # noqa: E402

POINTS = (0.475, 0.500, 0.700)
BUDGET = {round(r["g"], 3): r for r in json.load(open(paths.data("line_budget.json")))}
out = []
for g in POINTS:
    T, v = BUDGET[g]["nlo"]["T_p"], BUDGET[g]["v_GeV"]
    # TL's defaults for this model (h3 in the temporal sector, g3^2 at NLO, mu4 = 7.0553 T, mu3 = g T) and TL's own
    # path through the channels (nlo_tables.fast_row): the bounce in units of T, then every channel on it
    pot = nt.load_model("TL_conformal_u1_no_fermions_3d")({"g": g, "v_GeV": v}, verbose=False)
    p, S3, r, phi3, info = nt._bounce(pot, T)
    r_u, phi_u, m3sq, lam3, g3sq, mD2 = nt._in_units(p, r, phi3, T)
    h3 = nt._h3_in_units(pot, p, T)
    V3_GeV, Z3_GeV = pot.two_loop_potential(p), pot.two_loop_kinetic(p)
    V3n = lambda q: V3_GeV(q * T ** 0.5) / T ** 3  # noqa: E731
    Z3n = lambda q: Z3_GeV(q * T ** 0.5)  # noqa: E731
    a = abelian_higgs_3d.nlo_assembly(r_u, phi_u, m3sq, lam3, g3sq, mD2, S3, 0.0, T=1.0, V3_nlo=V3n, Z3_nlo=Z3n, h3=h3)
    mix = nt.mixing_row(pot, T, abelian_higgs_3d.MIXING_L_MAX)["value"]
    delta_S_grid = float(nt.correction_for_model(pot)(T, S3_LO=float(S3)))        # with the dimension-six shifts
    dS6, dSkin = nt._dim6(p, r, phi3, info)
    entry = dict(g=g, v_GeV=v, T_GeV=T, T_over_mu0=T / 1.0, m3sq=p["m3sq"], lam3=p["lam3"], g3sq=p["g3sq"], mD2=p["mD2"],
                 lambda_vl=p["lambda_vl"], S3_LO=float(S3),
                 radial=a["radial"], goldstone=a["goldstone_xi1"], gauge=a["gauge_single_channel"],
                 temporal=a["temporal"], half_mixing=0.5 * mix, j0_correction=a["j0_correction"],
                 double_counting_C=a["double_counting_C"], zero_mode_jacobian_log=a["zero_mode_jacobian_log"],
                 log_A_dyn_over_T=a["log_A_dyn_over_T"], nlo_potential=a["nlo_potential"], nlo_kinetic=a["nlo_kinetic"],
                 dim6_sextic=float(dS6), dim6_kinetic=float(dSkin), delta_S_grid=delta_S_grid,
                 delta_S_grid_without_dim6=delta_S_grid - float(dS6) - float(dSkin))
    # the rows of table 3: S_6 of eq. (A.4), the exponent E of eq. (A.2), and Delta S of eq. (A.3) solved directly
    # (fixed point in S_tot = S3/T + Delta S) instead of read off the grid
    entry["S6"] = float(dS6) + float(dSkin)
    entry["E"] = (entry["S3_LO"] + entry["radial"] + entry["goldstone"] + entry["gauge"] + entry["temporal"]
                  + entry["half_mixing"] + entry["j0_correction"] - entry["double_counting_C"] + entry["nlo_potential"]
                  + entry["S6"])
    dS = 0.0
    for _ in range(100):
        dS = entry["E"] - entry["S3_LO"] - entry["log_A_dyn_over_T"] + 1.5 * math.log((entry["S3_LO"] + dS) / (2 * math.pi))
    entry["delta_S"] = dS
    entry["three_halves_log_S_tot"] = 1.5 * math.log((entry["S3_LO"] + dS) / (2 * math.pi))
    out.append(entry)
    print(f"g = {g}: T = {T:.4g} GeV, S3 = {S3:.4f}, Delta S (grid) {delta_S_grid:.4f}, radial {a['radial']:.4f}, "
          f"goldstone {a['goldstone_xi1']:.4f}, gauge {a['gauge_single_channel']:.4f}, temporal {a['temporal']:.4f}, "
          f"C {a['double_counting_C']:.4f}, 1/2 mixing {0.5*mix:.4f}, J0 {a['j0_correction']:.4f}, "
          f"V3NLO {a['nlo_potential']:.4f}, kinetic {a['nlo_kinetic']:.4f}, lnAdyn/T {a['log_A_dyn_over_T']:.4f}", flush=True)
here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
json.dump(out, open(os.path.join(here, "determinant_reference_values.json"), "w"), indent=1)
keys = list(out[0].keys())
with open(paths.data("determinant_reference_values.csv"), "w") as f:
    f.write(",".join(keys) + "\n")
    for e in out:
        f.write(",".join(f"{e[k]:.10g}" if isinstance(e[k], float) else str(e[k]) for k in keys) + "\n")
print("written")
