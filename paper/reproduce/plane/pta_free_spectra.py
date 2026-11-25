"""Writes paper/data/pta_free_spectra.npz, the free spectra drawn as violins in figure fig:spectra: per data set the
frequencies, the grid of log10 rho, the log-density of every bin on that grid and the time span, read with binaryfog
(binaryfog.FreeSpectrum.load fetches the public releases): NG15 = ng15_cp_v1 (2024 NANOGrav release, as PTArcade),
NG15v2 = ng15_cp_v2 (corrected 2026 release, the default), IPTA2 = ipta2_cp_v1, with the bins of the fits.
    python pta_free_spectra.py [--check]"""
import os
import sys

import numpy as np

import binaryfog as bf  # pip install binaryfog==1.0.0

HERE = os.path.dirname(os.path.abspath(__file__))
SETS = (("NG15", "ng15_cp_v1", 14), ("NG15v2", "ng15_cp_v2", 14), ("IPTA2", "ipta2_cp_v1", 13))
out = {}
for key, name, nb in SETS:
    d = bf.FreeSpectrum.load(name, nbins=nb)
    out.update({f"{key}_freqs": d.freqs, f"{key}_rho_grid": d.grid, f"{key}_density": d.density,
                f"{key}_Tspan": np.array(d.tspan)})
path = os.path.join(HERE, "..", "..", "data", "pta_free_spectra.npz")
if "--check" in sys.argv:
    z = np.load(path)
    print("identical" if set(z.files) == set(out) and all(np.array_equal(z[k], out[k]) for k in out) else
          {k: float(np.max(np.abs(z[k] - out[k]))) for k in out})
else:
    np.savez(path, **out)
    print("wrote", path)
