"""Merge variant tables of the same (G, LOGV) grid into one exact table.

  python merge_tables.py OUT BASE EXTRA [EXTRA ...]

BASE is a table of build_exact_table.py; each EXTRA (for example the falsevac_table.npz of
build_falsevac_table.py, variant "fvbg") adds its variants along the variant axis of every
per-variant key. Keys without a variant axis (G, LOGV, T_reh_SM, T_reh_SM4) are taken from BASE.
A variant of an EXTRA that BASE already has replaces the one of BASE (printed), so that a table
of a few rebuilt variants can be merged into a full one. Refuses grids that differ. The paper's
table is paper/data/transition_table.npz, merged from <work>/exact_table.npz and
<work>/falsevac_table.npz."""
import sys

import numpy as np

out, base, extras = sys.argv[1], sys.argv[2], sys.argv[3:]
B = dict(np.load(base, allow_pickle=True))
names = [str(v) for v in B["VARIANTS"]]
for path in extras:
    E = dict(np.load(path, allow_pickle=True))
    assert np.array_equal(E["G"], B["G"]) and np.array_equal(E["LOGV"], B["LOGV"]), path
    evars = [str(v) for v in E["VARIANTS"]]
    replaced = [v for v in evars if v in names]
    new = [v for v in evars if v not in names]
    if replaced:
        print(f"{path}: replaces {replaced}")
    for k, v in E.items():
        if k in ("G", "LOGV", "VARIANTS") or v.ndim != 3:
            continue
        assert k in B and B[k].ndim == 3, (path, k)
        for j, var in enumerate(evars):
            if var in replaced:
                B[k][:, names.index(var), :] = v[:, j, :]
        if new:
            B[k] = np.concatenate([B[k], v[:, [evars.index(x) for x in new], :]], axis=1)
    # every per-variant key of the base must have received the new variants
    for k, v in B.items():
        if k not in ("G", "LOGV", "VARIANTS") and v.ndim == 3:
            assert v.shape[1] == len(names) + len(new), (path, k, v.shape)
    names += new
B["VARIANTS"] = np.array(names)
np.savez(out, **B)
print("wrote", out, "with variants", names)
