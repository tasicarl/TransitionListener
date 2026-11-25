"""The default NANOGrav likelihood (NG15, no foreground) in the slow tail of the plane figure on a grid much
finer than the plane scan's, with the machinery of plane_scan.py unchanged: tp_variants,
cosmo_inputs, make_spectrum, lnL. Writes OUT (default paper/data/tail_check/tail_patch.npz) with lnL and, with TAIL_BHB=1,
lnL_bhb (NG15 with the binary foreground profiled to convergence, as the scan's default row),
(beta/H)_Gamma, (beta/H)_RH and the amplitude bracket A = 0.06 [1 + 0.8 (1 - exp(beta^-1/2))]
of the expansion fit (eq. lv_amp), clipped at zero as in TransitionListener.
  TAIL_DATASET=IPTA2 selects the IPTA likelihood for lnL.
  python tail_fine.py G_LO G_HI DG LV_LO LV_HI DLV [OUT]   (rows in parallel, TAIL_WORKERS, default 28)"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402  (src/, models/ and the folders of paper/reproduce on sys.path)
import plane_scan as PS  # noqa: E402

g_lo, g_hi, dg, l_lo, l_hi, dl = map(float, sys.argv[1:7])
out = sys.argv[7] if len(sys.argv) > 7 else paths.data("tail_check", "tail_patch.npz")
G = np.round(np.arange(g_lo, g_hi + dg / 2, dg), 6)
L = np.round(np.arange(l_lo, l_hi + dl / 2, dl), 5)
c = PS.COSMO_VARIANTS[0]
#: the data set of lnL (TAIL_DATASET, default NG15; IPTA2 for an IPTA patch of the plane figure)
DATASET = os.environ.get("TAIL_DATASET", "NG15")


def row(i):
    g = G[i]
    r = np.full((4, len(L)), np.nan)
    for j, lv in enumerate(L):
        tpv = PS.tp_variants(g, lv)
        _, bh, alpha, bhv = tpv["nlo"]
        r[1, j], r[2, j] = bh, bhv
        if not (np.isfinite(bh) and np.isfinite(alpha)):
            continue
        reh = PS.cosmo_inputs(c, g, lv, alpha)
        if not np.isfinite(reh[0]):
            continue
        spec = PS.make_spectrum(c[1], reh, bh, alpha, bhv)
        if spec is None:
            continue
        r[0, j] = PS.lnL(DATASET, spec)
        if BHB:
            r[3, j] = PS.lnL("NG15", spec, smbhb=True, optimise=True)
    return i, r


BHB = os.environ.get("TAIL_BHB") == "1"


lnl = np.full((len(G), len(L)), np.nan); bG = lnl.copy(); bR = lnl.copy(); lnb = lnl.copy()
t0 = time.time()
from multiprocessing import get_context  # noqa: E402
with get_context("fork").Pool(int(os.environ.get("TAIL_WORKERS", "28"))) as pool:
    for i, r in pool.imap_unordered(row, range(len(G))):
        lnl[i], bR[i], bG[i], lnb[i] = r
A = np.where(bG > 0, np.maximum(0.0, 0.06 * (1 + 0.8 * (1 - np.exp(np.sqrt(1 / np.where(bG > 0, bG, np.nan)))))), np.nan)
np.savez(out, G=G, LOGV=L, lnL=lnl, lnL_bhb=lnb, beta_Gamma_H=bG, betaH_R=bR, A=A)
print(f"wrote {out}: {len(G)} x {len(L)} points in {time.time() - t0:.0f} s")
