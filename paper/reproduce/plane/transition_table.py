"""The transition parameters on the (g, log10 v) plane, for every variant of the rate.

Source: paper/data/transition_table.npz (or $TRANSITION_TABLE), written by rate/build_exact_table.py and
merged with the false-vacuum background variant of rate/build_falsevac_table.py by rate/merge_tables.py.
For each coupling g of build_exact_table.G (0.39 to 0.8, 0.005 apart from 0.39 to 0.44), each log10(v/GeV)
of build_exact_table.LOGV (-2.625 to 4.375 in steps of 0.0625) and each rate variant, the false-vacuum fraction
is integrated exactly on the self-consistent mean background
(rate/mean_background.average_background), with the actions of rate/compute_actions.py and
Delta S from TransitionListener's shipped grid.
At fixed g the exponent is a function of T/v, and the only other scale is the Planck mass:
ln(Gamma/H_V^4) shifts by -4 ln(v/v_line), which is why beta/H at fixed g depends on v.

Keys per variant: T_p (true-vacuum fraction 0.29, GeV), T_nuc (one bubble per Hubble
volume), T_1e, betaH_S (T dS/dT at T_p), betaH_R ((8 pi/f_perc)^{1/3}/(R_* H) with the mean
bubble separation R_* = n_b^{-1/3} at T_p), beta_Gamma_H (d ln Gamma/dt / H at T_p), H_p_over_HV
(the mean-background Hubble rate at T_p over H_V = (Delta V/3)^{1/2}/M_Pl),
dlnV_dt_over_H (d ln(a^3 P_false)/dt / H at T_p; negative when the physical false-vacuum
volume shrinks), alpha (Delta V/rho_rad(T_p), four dark degrees of freedom in the plasma
before the transition), S_p, betaH_S_nuc (T dS/dT at T_nuc), in_table (1 where S_LO at T_p
lies inside the range of the Delta S grid), window_limited (1 where a variant does not percolate inside the integration
window while ln(Gamma/H^4) still rises at its lowest temperature: inconclusive rather than no
transition). T_reh_SM and T_reh_SM4: rho_R(T_reh) = Delta V with the Standard Model alone
and with four dark degrees of freedom added.

Variants: those of build_exact_table.VARIANTS and fvbg (the false-vacuum background). Every
variant also has P_t_end, the largest true-vacuum fraction the window reaches (below f_perc where
the transition does not percolate).

Interpolation: PCHIP (shape-preserving cubic) in (g, log10 v) of the logarithm of every positive
quantity (linear keys as they are). Nodes where a variant does not percolate are filled with the
nearest value along log10 v for the interpolation, and whether a point percolates is decided by
the interpolated P_t_end >= f_perc (0.99 for T_f, the completion), so that the edges of the
region without a transition are smooth rather than following the cells of the node grid. The
quantity interpolated for this decision is ln I_end = ln(-ln(1 - P_t_end)), the logarithm of the
percolation integral at the end of the window, which is close to linear in log10 v (P_t_end itself
jumps from about 0.1 to 1 within two nodes); nodes where P_t_end = 1 to double precision are
extended from the nearest node below one with the slope d ln I/d log10 v = -4 ln 10 (at fixed g
the rate normalisation Gamma/H_V^4 scales as v^-4 at fixed T/v), and capped at ln I_end = LN_I_CAP. A node that percolates but lacks a quantity keeps it missing (NaN spreads
to its neighbourhood).
"""
import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data")      # paper/data
# TRANSITION_TABLE points a second instance of this module at another table of the same layout
# (load_instance below; used for the four-dimensional daisy rates of daisy/prepare_daisy_table.py)
# read into memory at import: a lazy NpzFile shares one file descriptor, and with it one file offset, with every
# forked worker, whose concurrent reads then corrupt each other ("Overlapped entries")
_D = dict(np.load(os.environ.get("TRANSITION_TABLE", os.path.join(DATA, "transition_table.npz"))))
G_TAB, LOGV_TAB = _D["G"], _D["LOGV"]
VARIANTS = [str(v) for v in _D["VARIANTS"]]
LINEAR = ("dlnV_dt_over_H", "in_table", "window_limited", "P_t_end", "f_window_limited")
F_PERC = 0.28957
P_FINAL = 0.99
LN_I_CAP = 12.0         # ln I_end where P_t_end = 1 cannot be resolved (I_end = 1.6e5)
LN_I_SLOPE = -4.0 * np.log(10.0)
KEYS = ("T_p", "T_nuc", "T_1e", "betaH_S", "betaH_R", "beta_Gamma_H", "dlnV_dt_over_H", "alpha",
        "S_p", "betaH_S_nuc", "in_table", "window_limited", "T_min_window", "T_f", "P_t_end", "f_window_limited",
        "H_p_over_HV")
_CACHE = {}


def _fill_along_v(y, fill):
    """y (g, logv) with the nodes `fill` replaced by the nearest-in-log10-v value of the same g
    that is not to be filled (rows without any are filled from the nearest such row in g)."""
    y = np.array(y, dtype=float)
    keep = ~fill & np.isfinite(y)
    for i in range(y.shape[0]):
        if keep[i].any():
            idx = np.flatnonzero(keep[i])
            for j in np.flatnonzero(~keep[i]):
                if fill[i, j] or not np.isfinite(y[i, j]):
                    if fill[i, j]:
                        y[i, j] = y[i, idx[np.argmin(np.abs(idx - j))]]
    rows = np.flatnonzero(keep.any(axis=1))
    for i in np.flatnonzero(~keep.any(axis=1)):
        y[i] = y[rows[np.argmin(np.abs(rows - i))]]
    return y


def _p_end(variant):
    v = "nlo" if variant is None else variant
    return np.array(_D["P_t_end"][:, VARIANTS.index(v), :], dtype=float)


def _threshold(key, variant):
    if key == "T_f":
        return P_FINAL
    return (1 - np.exp(-1.0)) if variant == "perc1e" else F_PERC


def _ln_i_end(variant):
    """ln(-ln(1 - P_t_end)) on the nodes, (g, logv): nodes with P_t_end = 1 extended from the
    nearest node below one with the slope LN_I_SLOPE, everything capped at LN_I_CAP; nodes without
    a value (the window failed) take the nearest value along log10 v."""
    pe = _p_end(variant)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.log(-np.log1p(-np.clip(pe, 0.0, 1.0)))
    z[np.isfinite(pe) & (pe <= 0.0)] = -50.0
    sat = np.isfinite(pe) & (pe >= 1.0)
    for i in range(z.shape[0]):
        fin = np.flatnonzero(np.isfinite(z[i]))
        for j in np.flatnonzero(sat[i]):
            if len(fin) == 0:
                z[i, j] = LN_I_CAP
                continue
            a = fin[np.argmin(np.abs(fin - j))]
            z[i, j] = z[i, a] + LN_I_SLOPE * (LOGV_TAB[j] - LOGV_TAB[a])
    z = np.minimum(z, LN_I_CAP)
    return _fill_along_v(z, ~np.isfinite(z))


def _interpolator(name, variant=None):
    key = (name, variant)
    if key not in _CACHE:
        y = _D[name] if (variant is None or _D[name].ndim == 2) else _D[name][:, VARIANTS.index(variant), :]
        y = np.array(y, dtype=float)
        if name not in LINEAR:
            y = np.where(y > 0, np.log(np.where(y > 0, y, 1.0)), np.nan)
        if name.startswith("T_reh"):
            _CACHE[key] = (RegularGridInterpolator((G_TAB, LOGV_TAB), y, method="pchip", bounds_error=False,
                                                   fill_value=np.nan), None, None)
            return _CACHE[key]
        pe = _p_end(variant)
        node_ok = np.nan_to_num(pe, nan=0.0) >= _threshold(name, variant)
        missing = node_ok & ~np.isfinite(y)          # percolates, but the quantity is not defined
        if not np.isfinite(y).any():
            _CACHE[key] = (lambda pts: np.full(len(pts), np.nan), None, None)
            return _CACHE[key]
        y = _fill_along_v(y, ~np.isfinite(y) | ~node_ok)
        smooth = RegularGridInterpolator((G_TAB, LOGV_TAB), y, method="pchip", bounds_error=False, fill_value=np.nan)
        pe_int = RegularGridInterpolator((G_TAB, LOGV_TAB), _ln_i_end(variant), method="pchip",
                                         bounds_error=False, fill_value=np.nan)
        thr = float(np.log(-np.log1p(-_threshold(name, variant))))
        if missing.any():
            miss_int = RegularGridInterpolator((G_TAB, LOGV_TAB), missing.astype(float), method="linear",
                                               bounds_error=False, fill_value=np.nan)
            mask = lambda pts, pe_int=pe_int, miss_int=miss_int, thr=thr: np.where(miss_int(pts) > 0, -np.inf, pe_int(pts))
        else:
            mask = pe_int
        _CACHE[key] = (smooth, mask, thr)
    return _CACHE[key]


def get(key, g, logv, variant="nlo"):
    """key of variant at (g, log10 v); arrays broadcast. NaN where the variant does not
    percolate (for T_f: does not complete) at that point."""
    g, logv = np.broadcast_arrays(np.asarray(g, float), np.asarray(logv, float))
    pts = np.stack([g.ravel(), logv.ravel()], axis=-1)
    smooth, mask, thr = _interpolator(key, variant)
    y = smooth(pts)
    if mask is not None:
        y = np.where(mask(pts) >= thr, y, np.nan)
    y = y.reshape(g.shape)
    return y if key in LINEAR else np.exp(y)


def margin(key, g, logv, variant="nlo"):
    """The interpolated ln I_end minus its value at the threshold of `key` (f_perc, or 0.99 for
    T_f): positive where `get` returns a value, negative where it returns NaN. A continuous field
    whose zero contour is the edge of the region without a transition."""
    g, logv = np.broadcast_arrays(np.asarray(g, float), np.asarray(logv, float))
    pts = np.stack([g.ravel(), logv.ravel()], axis=-1)
    _, mask, thr = _interpolator(key, variant)
    return (mask(pts) - thr).reshape(g.shape)


_TL = {}


def reheating_tl(g, logv):
    """(T_reh in GeV, g_eff, h_eff) right after reheating, from TransitionListener at the nodes
    (tl_nodes.py, collected by collect_tl_nodes.py into tl_nodes_table.npz), PCHIP in
    (g, log10 v) of the logarithms. Defined everywhere in the table: the reheating temperature
    follows from the vacuum energy and the degrees of freedom, not from whether the transition
    completes."""
    if "T_reh_TL" not in _TL:
        path = os.path.join(DATA, "tl_nodes_table.npz")
        if not os.path.exists(path):
            raise FileNotFoundError("tl_nodes_table.npz is missing: run tl_nodes.py and collect_tl_nodes.py")
        N = np.load(path)
        for key in ("T_reh_TL", "g_reh_TL", "h_reh_TL"):
            _TL[key] = RegularGridInterpolator((N["G"], N["LOGV"]), np.log(N[key]), method="pchip",
                                               bounds_error=False, fill_value=np.nan)
    g, logv = np.broadcast_arrays(np.asarray(g, float), np.asarray(logv, float))
    pts = np.stack([g.ravel(), logv.ravel()], axis=-1)
    return tuple(np.exp(_TL[k](pts)).reshape(g.shape) for k in ("T_reh_TL", "g_reh_TL", "h_reh_TL"))


def kappa_tl(g, logv):
    """The efficiency kappa = kappa_phi + kappa_sw at (g, log10 v), from TransitionListener at the nodes (kappa_nodes_table.npz),
    PCHIP in (g, log10 v) of ln kappa. A node without a finished run takes the value of its coupling along log10 v."""
    if "kappa" not in _TL:
        N = np.load(os.path.join(DATA, "kappa_nodes_table.npz"))
        y = np.log(N["kappa"])
        for i in range(y.shape[0]):
            ok = np.isfinite(y[i])
            y[i] = np.interp(N["LOGV"], N["LOGV"][ok], y[i][ok])
        _TL["kappa"] = RegularGridInterpolator((N["G"], N["LOGV"]), y, method="pchip", bounds_error=False, fill_value=None)
    g, logv = np.broadcast_arrays(np.asarray(g, float), np.asarray(logv, float))
    return np.exp(_TL["kappa"](np.stack([g.ravel(), logv.ravel()], axis=-1))).reshape(g.shape)


def t_reh(g, logv, dark_dof=0.0):
    """Reheating temperature in GeV: the Standard Model alone (dark_dof = 0) or with four
    dark degrees of freedom in the plasma (dark_dof = 4)."""
    if dark_dof not in (0.0, 4.0):
        raise ValueError("the table has T_reh for 0 and 4 dark degrees of freedom only")
    g, logv = np.broadcast_arrays(np.asarray(g, float), np.asarray(logv, float))
    smooth, _, _ = _interpolator("T_reh_SM" if dark_dof == 0.0 else "T_reh_SM4")
    y = smooth(np.stack([g.ravel(), logv.ravel()], axis=-1)).reshape(g.shape)
    return np.exp(y)


def load_instance(path, name="transition_table_instance"):
    """A separate instance of this module that reads the table at `path` (same layout, its own
    G grid and variants), for get() and margin() on that table."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, os.path.abspath(__file__))
    mod = importlib.util.module_from_spec(spec)
    old = os.environ.get("TRANSITION_TABLE")
    os.environ["TRANSITION_TABLE"] = path
    try:
        spec.loader.exec_module(mod)
    finally:
        if old is None:
            del os.environ["TRANSITION_TABLE"]
        else:
            os.environ["TRANSITION_TABLE"] = old
    return mod
