"""EFT validity at the percolation temperature over the preferred regions: the ratios of
transitionlistener.eft_rate.abelian_higgs_3d.validity_ratios (heaviest kept boson over 2 pi T along the bounce,
Debye mass over 2 pi T, 3D loop parameter g3^2/(pi m3), dim-6 shift over S3/T) at the NLO
percolation temperature of transition_table.npz, at the table nodes inside the 68% and 95%
regions of NANOGrav and IPTA, against the thresholds of validity.thresholds_for('soft', 'NLO')."""
import math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402  (src/, models/ and the folders of paper/reproduce on sys.path)
sys.path.insert(0, os.path.join(paths.PAPER, "figs"))
from transitionlistener.eft_rate import abelian_higgs_3d
from transitionlistener.eft_rate import validity
import dr_conformal_u1_no_fermions_matching as dr
import make_figures as mf
from scipy.interpolate import RegularGridInterpolator
T = np.load(paths.data("transition_table.npz"), allow_pickle=True)
k = [str(x) for x in T["VARIANTS"]].index("nlo")
lnTp = RegularGridInterpolator((T["G"], T["LOGV"]), np.log(T["T_p"][:, k, :]), bounds_error=False)
S3T = RegularGridInterpolator((T["G"], T["LOGV"]), T["S_p"][:, k, :], bounds_error=False) if "S_p" in T.files else None
d = dict(np.load(paths.data("plane_scan_u1.npz"), allow_pickle=True))
kd = [str(n) for n in d["names"]].index("nlo")
Gm, Lm = np.meshgrid(d["G"], d["LOGV"], indexing="ij")
thr = validity.thresholds_for("soft", "NLO")
print("thresholds:", {key: v[:2] for key, v in thr.items()})
for ds in ("NG15", "IPTA2"):
    lv, p = mf._levels(d[ds][kd])
    for lab, reg in (("68%", p >= lv[1]), ("95%", p >= lv[0])):
        idx = np.argwhere(reg)
        sel = idx[np.linspace(0, len(idx) - 1, min(60, len(idx))).astype(int)]
        out = []
        for i, j in sel:
            g, logv = float(Gm[i, j]), float(Lm[i, j])
            Tp = float(np.exp(lnTp([[g, logv]])[0]))
            mu0 = g * 10**logv * math.exp(-1 / 6)
            q = dr.matched_soft_parameters(g, 0.0, mu0, Tp, mu4_over_T=7.0553, matching_order="NLO", mu3=g * Tp)
            S, r, phi3, _ = abelian_higgs_3d.solve_LO_bounce(q["m3sq"], q["lam3"], q["g3sq"], m_debye_sq=q["mD2"], wall_power=12)
            rat = abelian_higgs_3d.validity_ratios(Tp, q["m3sq"], q["lam3"], q["g3sq"], q["mD2"], r, phi3,
                                        fermion_masses_over_field=(), c6=q["c6"], S3_over_T=S)   # solve_LO_bounce returns the dimensionless exponent S3/T
            out.append([rat["hard"], rat["debye"], rat["expansion_3d"], rat["dim6"] if rat["dim6"] is not None else np.nan, g])
        out = np.array(out)
        print(f"{ds} {lab} ({len(out)} nodes): hard {out[:,0].min():.3f}-{out[:,0].max():.3f}, debye {out[:,1].min():.3f}-{out[:,1].max():.3f}, "
              f"expansion_3d {out[:,2].min():.3f}-{out[:,2].max():.3f}, dim6 {np.nanmin(out[:,3]):.2e}-{np.nanmax(out[:,3]):.2e}")
