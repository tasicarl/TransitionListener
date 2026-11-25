"""The six fits without expansion of 2208.11697 with and without the causal f^3 tail (footnote on the causal factor,
section 3.3): NANOGrav and IPTA maps on the plane grid, written to paper/data/notail_scan.npz.

    PYTHONPATH=<repo>/src python notail_scan.py WORKERS        
The rows with the tail repeat rows of plane_scan_u1.npz and are compared with them as a check."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402  (src/, models/ and the folders of paper/reproduce on sys.path)
workers = int(sys.argv[1])
os.environ["PLANE_SCAN_ARGS"] = "--out " + paths.data("notail_scan.npz")
import numpy as np  # noqa: E402
import plane_scan as PS  # noqa: E402
from multiprocessing import get_context  # noqa: E402

FLAT = [c for c in PS.COSMO_VARIANTS if c[0].startswith("no expansion")]
PS.VARIANT_LIST[:] = ([("cosmology", c[0], "nlo", c) for c in FLAT]
                      + [("cosmology", c[0] + ", no tail", "nlo", (c[0] + ", no tail", dict(c[1], tail="off"))) for c in FLAT])
names = [v[1] for v in PS.VARIANT_LIST]
keys = [ds for ds in PS.DATASETS]
new = {k: np.full((len(names), len(PS.G), len(PS.V)), np.nan) for k in keys}
t0 = time.time()
with get_context("fork").Pool(min(workers, len(PS.G))) as pool:
    for n, (ig, o, e, p) in enumerate(pool.imap_unordered(PS.row, range(len(PS.G)))):
        for k in keys:
            new[k][:, ig, :] = o[k]
        if n % 20 == 0:
            print(f"{n + 1}/{len(PS.G)} couplings done, {time.time() - t0:.0f} s", flush=True)
np.savez(PS.ARGS.out, G=PS.G, LOGV=PS.LOGV, names=np.array(names), **new)
F = np.load(paths.data("plane_scan_u1.npz"), allow_pickle=True)
fn = list(F["names"])
for c in FLAT:
    for k in keys:
        a, b = F[k][fn.index(c[0])], new[k][names.index(c[0])]
        print(f"check {c[0]} {k}: max |difference| to plane_scan_u1 {np.nanmax(np.abs(a - b)):.3g}", flush=True)
print("wrote", PS.ARGS.out)
