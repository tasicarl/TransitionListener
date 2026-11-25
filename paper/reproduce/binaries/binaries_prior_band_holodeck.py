"""The binaryfog prior prediction of figure 3 on the whole frequency axis, 0.8 to 100 nHz, from holodeck itself rather
than the emulator, whose library grids end at 1.05 and 32.6 nHz. Draws from a prior set of binaryfog.priors
(default 'nanograv', the NANOGrav priors of table tab:binpriors; 'recent' the survey-averaged ones; truncated to the
library box), one holodeck call per draw with the
model of the binaryfog emulator library (PS_Classic_Phenom_Uniform), on 24 log-spaced bin
centres, each bin 1/T_IPTA = 1.0496 nHz wide as the IPTA DR2 bins (the width sets the scatter between realisations).
Per draw 400 realisations; 100 of them are kept, so that the quantiles pool the prior and the realisations.
    (an environment with holodeck and binaryfog 1.0.0)  python binaries_prior_band_holodeck.py [NDRAW] [JOBS] [PRIOR]   -> paper/data/binaries_prior_band_holodeck_<PRIOR>.npz"""
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import multiprocessing as mp  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
import holodeck as holo  # noqa: E402
import holodeck.librarian  # noqa: E402,F401  (not imported by holodeck itself)
from holodeck import cosmo  # noqa: E402
from holodeck.sams import sam_cyutils  # noqa: E402
from binaryfog import priors as bfp  # noqa: E402  (pip install binaryfog==1.0.0)
from binaryfog.background import PARAM_NAMES, PRIOR_BOX  # noqa: E402

NDRAW = int(sys.argv[1]) if len(sys.argv) > 1 else 240
JOBS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
PRIOR = sys.argv[3] if len(sys.argv) > 3 else "nanograv"     # binaryfog.priors set
# the paper's names of the population priors and their binaryfog.priors sets
BF_SET = {"literature": "tomczak_kormendy", "recent": "survey_average"}.get(PRIOR, PRIOR)
NREAL, NKEEP = 400, 100
STALL = 900
FLOOR = -60.0                                     # log10 h_c^2 of an empty bin
F = np.geomspace(0.8e-9, 1.0e-7, 24)
DF = 1.0496e-9
EDGES = (F - 0.5 * DF, F + 0.5 * DF)
H100 = 100 / 3.0857e19

_SPACE = None


def _get_space():
    """holodeck's parameter space PS_Classic_Phenom_Uniform, the model of the binaryfog 1.0.0 emulator library
    (adapted from the script that built that library)."""
    global _SPACE
    if _SPACE is None:
        _SPACE = holo.librarian.param_spaces_dict["PS_Classic_Phenom_Uniform"]()
    return _SPACE


def draws(rng, n):
    gauss = bfp.GAUSSIAN_SETS[BF_SET]
    lo = np.array([PRIOR_BOX[k][0] for k in PARAM_NAMES]); hi = np.array([PRIOR_BOX[k][1] for k in PARAM_NAMES])
    out = []
    while sum(len(x) for x in out) < n:
        th = rng.uniform(lo, hi, size=(4 * n, len(PARAM_NAMES)))
        for k, name in enumerate(PARAM_NAMES):
            if name in gauss:
                th[:, k] = rng.normal(*gauss[name], size=len(th))
        out.append(th[np.all((th >= lo) & (th <= hi), axis=1)])
    return np.concatenate(out)[:n]


def one(job):
    i, theta = job
    np.random.seed(770000 + i)
    try:
        sam, hard = _get_space().model_for_params(dict(zip(PARAM_NAMES, map(float, theta))))
        res = []
        for fc, lo, hi in zip(F, *EDGES):                  # one bin at a time: the bins overlap at low f
            raw = np.array([fc]) / 2.0
            edges_f = np.array([max(lo, 1e-12), hi]) / 2.0
            redz, dnum = sam_cyutils.dynamic_binary_number_at_fobs(raw, sam, hard, cosmo)
            edges = [sam.mtot, sam.mrat, sam.redz, edges_f]
            number = sam_cyutils.integrate_differential_number_3dx1d(edges, np.ascontiguousarray(dnum))
            rz = np.ascontiguousarray(redz); rz[rz < 0.0] = -np.inf
            gwb = holo.gravwaves._gws_from_number_grid_integrated_redz(edges, rz, number, NREAL)
            res.append(2 * np.log10(np.asarray(gwb)).ravel())
        hc2 = np.array(res)                                  # (nfreq, NREAL), log10 h_c^2
        keep = np.random.choice(NREAL, NKEEP, replace=False)
        return i, hc2[:, keep]
    except Exception as exc:                                  # noqa: BLE001
        print(f"  draw {i}: {type(exc).__name__}: {exc}", flush=True)
        return i, None


if __name__ == "__main__":
    t0 = time.time()
    th = draws(np.random.default_rng(20261109), NDRAW)
    out = np.full((NDRAW, len(F), NKEEP), np.nan)
    # a draw that has not returned STALL seconds after the last one did is recorded as hung and left out
    pool = mp.Pool(JOBS)
    pending = {i: pool.apply_async(one, ((i, t),)) for i, t in enumerate(th)}
    last, n = time.time(), 0
    while pending and time.time() - last < STALL:
        for i in [i for i, a in pending.items() if a.ready()]:
            r = pending.pop(i).get()[1]
            n += 1; last = time.time()
            if r is not None:
                out[i] = r
            if n % 20 == 0:
                print(f"  {n}/{NDRAW}, {(time.time() - t0) / 60:.1f} min", flush=True)
        time.sleep(1)
    hung = sorted(pending)
    pool.terminate()
    for i in hung:
        print(f"  draw {i} hung (no result {STALL} s after the last draw): theta = {np.round(th[i], 4).tolist()}", flush=True)
    # realisations without a binary in a bin (h_c = 0, log10 = -inf) are kept at a floor far below every quantile,
    # so that draws whose binaries stall still count; only draws that failed or hung (all nan) are left out
    ok = ~np.all(np.isnan(out), axis=(1, 2))
    if ok.sum() < 0.9 * NDRAW:
        raise SystemExit(f"only {ok.sum()} of {NDRAW} draws returned a spectrum")
    out = np.where(np.isneginf(out), FLOOR, out)
    assert np.all(np.isfinite(out[ok]))
    # h^2 Omega = 2 pi^2 f^2 h_c^2 / (3 H100^2)
    lom = out[ok] + np.log10(2 * np.pi ** 2 * F[None, :, None] ** 2 / (3 * H100 ** 2))
    pooled = np.moveaxis(lom, 1, 0).reshape(len(F), -1)
    Q = np.array([0.02275, 0.15866, 0.5, 0.84134, 0.97725])
    np.savez(os.path.join(HERE, "..", "..", "data", f"binaries_prior_band_holodeck_{PRIOR}.npz"), f=F, quantiles=Q,
             log10_h2omega=np.quantile(pooled, Q, axis=1), ndraw_ok=int(ok.sum()), theta=th, prior=PRIOR,
             hung=np.array(hung, int))
    print(f"done: {ok.sum()} of {NDRAW} draws, {(time.time() - t0) / 60:.1f} min", flush=True)
