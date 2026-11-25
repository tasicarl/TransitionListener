"""Joins the extension of the tail patches to 100 GeV (<work>/*_high.npz, log10 v = 1.305 to 2.0, the second part
of reproduce/jobs/09_tail.sh) to the patches up to log10 v = 1.3, in place. The cells up to 1.3 are copied unchanged; the no-signal values (plateau_*) do not depend on the grid."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "tail_check")
for name in ("tail_patch", "tail_fog", "tail_fog_recent"):
    a = dict(np.load(os.path.join(HERE, name + ".npz")))
    b = np.load(paths.work(name + "_high.npz"))
    n = len(a["LOGV"])
    if a["LOGV"][-1] > 1.31:
        print(name, "already joined")
        continue
    assert np.array_equal(a["G"], b["G"]) and np.allclose(b["LOGV"][0] - a["LOGV"][-1], 0.005, atol=1e-6)
    out = {}
    for k, x in a.items():
        out[k] = np.concatenate([x, b[k]], axis=-1) if x.ndim and x.shape[-1] == n else x
    np.savez(os.path.join(HERE, name + ".npz"), **out)
    print(name, "joined:", out["LOGV"][0], out["LOGV"][-1], len(out["LOGV"]))
