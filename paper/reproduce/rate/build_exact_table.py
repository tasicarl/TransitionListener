"""Transition parameters on a (g, log10 v) grid for every rate variant, by exact percolation.

For each coupling and v: T_p, T_nuc, T at P_false = 1/e, (beta/H)_S, (beta/H)_R, beta_Gamma/H,
d ln(a^3 P_f)/dt / H at T_p, alpha(T_p), (beta/H)_S at T_nuc, and T_reh for the SM alone and
for the SM plus the four dark degrees of freedom (rho_R(T_reh) = Delta V). Variants of the rate:
  nlo       S3 (NLO matching) + Delta S (mu4 x 1) with S_6 inside E, the first-order shift of the two
            dimension-six operators that act on the bounce, (Phi^+Phi)^3 and (Phi^+Phi)(D_i Phi)^+(D_i Phi), on the
            LO bounce (eq. A.3); Delta S then solves the fixed point of eq. (A.4) with S_6 in E and in S_tot, as
            TransitionListener does (nlo_tables._with_dim6). Every variant below carries S_6 of its own action kind
            (the operators come from the hard matching, so the LO-bounce variants keep them too, added to the bare
            action since those variants have no prefactor), except dim4. EXACT_DIM6=0 drops S_6 everywhere.
  mu05/mu2  S3 and Delta S at mu4 x 0.5 / x 2
  next+/-   nlo -/+ x3(T) max(|Delta S(T)|, 1)  (x3 = g3^2/(pi m3); the floor avoids the zero
            crossing of Delta S near g = 0.55)
  hord+/-   nlo -/+ x3(T) |S_NLO(T) - S_LO(T)|: the next order estimated from the whole step from LO
            to NLO (matching and Delta S), scaled by the 3D loop expansion parameter
  latc+/-   nlo -/+ 0.2 S_nlo(anchor): a constant shift of a fifth of the exponent at the line's
            percolation temperature, the size of the lattice-perturbation discrepancy of 2404.01876
            (20% in |ln Gamma|); an illustration of the reach of the data, not an uncertainty
  latp+/-   nlo x (1 -/+ 0.2): the same fifth applied to the exponent at every temperature
            Only the "-" variants (a smaller rate, the direction the lattice found) are built: the "+"
            variants percolate at the hottest temperatures of the grid (window_limited by HOT_MARGIN).
  dim4      nlo without Delta S_dim6: the 3D EFT truncated at dimension four
  adynT     A_dyn = T instead of Langer: S - 2.36
  pref+/-   the prefactor a factor 10 larger / smaller: S -/+ ln 10
  lorate    the LO bounce on the NLO-matched theory
  fulllo    the LO bounce on the LO-matched theory
  perc1e    nlo, percolation at P_false = 1/e
  dark      nlo, a ~ 1/T (a dark sector with its own temperature)
  mu3x05/mu3x2  the EFT scale mu3 = g T halved / doubled, consistently at this order: the part of the
            ln mu3 dependence of the LO bounce that the two-loop potential in Delta S does not cancel
  dStab     nlo with Delta S from the tables of the line scan (_RatioCorr, continued linearly in ln T outside each
            table)
Delta S: from the grid TransitionListener ships for this model (tab_data/nlo_grids/conformal_u1_no_fermions.npz, nlo_tables.py),
r = Delta S/S_LO as a function of S_LO = S_3/T at 30 couplings from g = 0.38 to 1, for mu4 x 0.5, 1 and 2, as
TransitionListener evaluates it, so that TransitionListener and these tables share one Delta S. The tables of the
line scan cover a range of temperatures around each line point and would have to be continued far outside it at
about a fifth of the nodes of the plane. EXACT_DS=tables makes the tables the default: they exist from g = 0.40 (G_TABLE_MIN); below,
the g = 0.40 table is used in units of the percolation temperature, and between line-scan couplings the neighbouring
tables are blended (_Blend).
in_table = 1 where S_LO at T_p lies inside the range of the grid (of the table with EXACT_DS=tables).
Every coupling's action grid includes the extensions of compute_actions.py (--low, --high, --higher) where they
exist in paths.ACTION_CACHE: at large v the transition percolates below the main grid at g = 0.44 to 0.49, and at
small v the slow couplings percolate above it. EXACT_VARIANTS restricts the
variants (comma-separated). A variant that percolates with an exponent within HOT_MARGIN of its value
at the hottest temperature of the grid is window-limited as well: the rate above the grid is then not
negligible and T_p is set by where the grid starts (the lattice-size "+" variants).
window_limited = 1 where a variant does not percolate inside the window but ln(Gamma/H^4) still
rises at its lowest temperature, or percolates within a factor 1.5 of it (inconclusive), 0
otherwise; T_min_window is that temperature. T_f is TransitionListener's completion temperature
(P_true = 0.99), NaN if not reached in the window; f_window_limited = 1 where that is because the
window ends while the rate still rises.
Writes <work>/exact_table.npz (or $EXACT_TABLE_OUT), which merge_tables.py combines with the false-vacuum table."""
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402  (src/, models/ and the folders of paper/reproduce on sys.path)
import exact_percolation as EP  # noqa: E402

# the lowest coupling with a Delta S table
G_TABLE_MIN = 0.40

# half steps from 0.395 to 0.435 and log10 v in steps of 0.0625, so that the slow tail of the preferred
# regions ((beta/H) < 4 at g = 0.40 to 0.44) is resolved by the nodes (no 0.445: the Delta S tables exist only
# at the couplings of the line scan, and from 0.44 up each coupling uses its own); 0.535 to 0.75 between the
# line-scan couplings above 0.52, with Delta S of the tables interpolated between the neighbours (class _Blend)
G = [0.39, 0.395, 0.40, 0.405, 0.41, 0.415, 0.42, 0.425, 0.43, 0.435, 0.44, 0.45, 0.46, 0.47, 0.475, 0.48, 0.49,
     0.5, 0.52, 0.535, 0.55, 0.5625, 0.575, 0.5875, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.8]
# v from 0.0024 GeV to 24 TeV, so that the posterior of every alternative lies inside the scanned range. At fixed g
# the actions depend on T/v alone, so the same caches serve every v.
LOGV = np.round(np.arange(-2.625, 4.376, 0.0625), 4)
VARIANTS = ["nlo", "mu05", "mu2", "next+", "next-", "dim4", "adynT", "pref+", "pref-", "lorate", "fulllo", "perc1e", "dark",
            "dStab", "mu3x05", "mu3x2", "hord+", "hord-",
            "latc-", "latp-"]
DIM6 = os.environ.get("EXACT_DIM6", "1") != "0"      # the dimension-six shifts in every variant but dim4
DS_SOURCE = os.environ.get("EXACT_DS", "grid")       # Delta S from TL's shipped grid, or "tables" (the line scan)
assert DS_SOURCE in ("grid", "tables"), DS_SOURCE
MU4_FACTORS = {"1": 1.0, "0.5": 0.5, "2": 2.0}
SUBSAMPLE = int(os.environ.get("EXACT_SUBSAMPLE", "8"))   # 1: the cache temperatures only (see refine)
if os.environ.get("EXACT_VARIANTS"):
    VARIANTS = [v for v in os.environ["EXACT_VARIANTS"].split(",") if v]
HOT_MARGIN = 10.0     # exponent units: the rate at the top of the grid is then below e^-10 of the rate at T_p
                      # (a margin of 2 is too lax: at S_hot - S_p = 4.6, (beta/H)_RH is 22% too large)
# H_p_over_HV: the mean-background Hubble rate at T_p in units of H_V = (Delta V/3)^{1/2}/M_Pl, so that
# beta/H_V = beta_Gamma_H * H_p_over_HV is d ln Gamma/dt at T_p in units of H_V (the speed of the
# expansion factors of arXiv:2511.15687). KEYS[:-5] are copied from percolate().
KEYS = ["T_p", "T_nuc", "T_1e", "betaH_S", "betaH_R", "beta_Gamma_H", "dlnV_dt_over_H", "alpha", "S_p", "T_f", "P_t_end",
        "H_p_over_HV", "betaH_S_nuc", "in_table", "window_limited", "T_min_window", "f_window_limited"]


def refine(T, S, sub=None):
    """The exponent S on the cache temperatures T, with sub - 1 temperatures added in every interval between finite
    rows (equally spaced in ln T) and S there from a PCHIP in ln T through the finite rows. percolate() samples the rate
    and the clock (g_s) only where the exponent is given and interpolates ln Gamma linearly between: on the cache alone
    (spacing 0.036 in ln T) (beta/H)_R is off by up to 1.3 % where beta/H ~ 40 to 140 at T_p ~ 0.1 to 0.25 GeV
    (g >= 0.6), mostly from the clock across the step of g_s at 0.12 GeV; with sub = 8 it agrees with sub = 32 to
    4e-4."""
    sub = SUBSAMPLE if sub is None else sub
    T, S = np.asarray(T, dtype=float), np.asarray(S, dtype=float)
    fin = np.isfinite(S)
    if sub <= 1 or fin.sum() < 2:
        return T, S
    from scipy.interpolate import PchipInterpolator
    x = np.log(T)
    p = PchipInterpolator(x[fin], S[fin])
    Tn, Sn = [T[:1]], [S[:1]]
    for i in range(len(T) - 1):
        if fin[i] and fin[i + 1]:
            xs = x[i] + (x[i + 1] - x[i]) * np.arange(1, sub) / sub
            Tn.append(np.exp(xs)); Sn.append(p(xs))
        Tn.append(T[i + 1:i + 2]); Sn.append(S[i + 1:i + 2])
    return np.concatenate(Tn), np.concatenate(Sn)


def sample(rm, T, S):
    """Hand the exponent S on the line temperatures T to the rate model of exact_percolation, refined (refine)."""
    Ts, Sr = refine(T, S)
    rm.Ts, rm.S3 = Ts, Sr
    rm.exponent = (lambda T_line, variant, Ts=Ts, Sr=Sr: np.interp(np.log(T_line), np.log(Ts), Sr,
                                                                    left=np.nan, right=np.nan))


class _RatioCorr:
    """Delta S from its table as the ratio r = Delta S/(S_3/T) of the table (S_3 of the NLO-matched LO
    bounce at the same temperatures), interpolated by PCHIP in ln T with its own end slopes, and continued
    linearly in ln T beyond the table with the slope at its end: Delta S = r (S_3/T) with the live S_3/T.
    The ratio and its slope are continuous everywhere.

    abelian_higgs_3d.NLOCorrection is not used here: it interpolates Delta S itself and takes its end slopes from the
    finite differences of the tabulated S_3, so its slope jumps at the edges where it meets r S_3 of the live
    S_3 (single-node spikes of up to 30% in (beta/H)_Gamma). Holding r constant beyond the table would switch
    off its temperature dependence at the edges, by about S_3 dr/dlnT ~ 0.2 in (beta/H)_Gamma, so that nodes
    just inside and just outside a table would disagree."""

    def __init__(self, T_GeV, delta_S, S3_LO):
        from scipy.interpolate import PchipInterpolator
        T, d, s3 = (np.asarray(a, dtype=float) for a in (T_GeV, delta_S, S3_LO))
        o = np.argsort(T)
        self._x, r = np.log(T[o]), d[o] / s3[o]
        self._r = PchipInterpolator(self._x, r)
        self._slope = self._r.derivative()(self._x[[0, -1]])
        self.T_range = (float(T[o][0]), float(T[o][-1]))

    def __call__(self, T_GeV, S3_LO=None):
        x = np.log(np.asarray(T_GeV, dtype=float))
        r = np.asarray(self._r(np.clip(x, self._x[0], self._x[-1])), dtype=float)
        r = (r + np.where(x < self._x[0], self._slope[0] * (x - self._x[0]), 0.0)
             + np.where(x > self._x[-1], self._slope[1] * (x - self._x[-1]), 0.0))
        out = r * np.asarray(S3_LO, dtype=float)
        return float(out) if out.ndim == 0 else out


class _GridCorr:
    """Delta S of the grid TransitionListener ships, at the coupling g and the mu4 factor, as the model's actionNuc
    evaluates it (nlo_tables.GridCorrection): r = Delta S/S_LO by PCHIP in ln S_LO, by a cubic spline in g. Outside the
    grid's range of S_LO (5 to 350) r is continued with the bounded slope of nlo_tables._continued; TL refuses those
    temperatures unless the rate there is shown not to matter, here in_table records whether T_p is inside."""

    def __init__(self, g, factor, mu4_over_T):
        from transitionlistener.eft_rate import nlo_tables
        grid = nlo_tables.load_grid(os.path.join(nlo_tables.GRID_DIR, "conformal_u1_no_fermions.npz"))
        # the conventions of the action caches (compute_actions.py): zero quartic at mu0, mu3 = g T, NLO matching
        assert grid.header["mu3_choice"] == "gT" and grid.header["input_scheme"] == "zero_quartic" \
            and grid.header["matching_order"] == "NLO" and abs(grid.header["mu4_over_T"] - mu4_over_T) < 1e-9, grid.header
        self._c = grid.correction({"g": g}, factor, hot_side_negligible=True, cold_side_negligible=True)
        self.S_range = self._c.S_range

    def __call__(self, T_GeV, S3_LO=None):
        return self._c(T_GeV, S3_LO=S3_LO)


class _Blend:
    """Delta S of a coupling g between two couplings of the line scan, ga < g < gb, which have
    tables: both tables in units of their percolation temperatures (T at g maps to T s_a and T s_b,
    s = anchor_neighbour/anchor_g), weighted linearly in g. Range: the overlap of the two."""

    def __init__(self, ca, cb, sa, sb, w):
        self.ca, self.cb, self.sa, self.sb, self.w = ca, cb, sa, sb, w
        self.T_range = (max(ca.T_range[0] / sa, cb.T_range[0] / sb), min(ca.T_range[1] / sa, cb.T_range[1] / sb))

    def __call__(self, T_GeV, S3_LO=None):
        T = np.asarray(T_GeV, dtype=float)
        out = (1 - self.w) * self.ca(T * self.sa, S3_LO=S3_LO) + self.w * self.cb(T * self.sb, S3_LO=S3_LO)
        return float(out) if np.ndim(out) == 0 else out


def with_S6(S3, dS, S6):
    """S3 + Delta S with S_6 inside the exponent E of eq. (A.2): Delta S = Delta S_0 + S_6 + 3/2 ln[(S3 + Delta S)/
    (S3 + Delta S_0)], solved by nlo_tables._with_dim6, where Delta S_0 = dS is the correction without S_6. Entries
    without a positive action keep S3 + dS + S6."""
    from transitionlistener.eft_rate.nlo_tables import _with_dim6
    S3, dS, S6 = (np.asarray(x, dtype=float) for x in (S3, dS, S6))
    out = S3 + dS + S6
    ok = np.isfinite(out) & (S3 > 0) & (S3 + dS > 0) & (out > 0) & (S6 != 0)
    if ok.any():
        out[ok] = S3[ok] * (1.0 + _with_dim6(S3[ok], dS[ok] / S3[ok], S6[ok] / S3[ok]))
    return out


class Model:
    """Rate model of one coupling from the action caches of compute_actions.py."""

    def __init__(self, g, blend=None):
        from transitionlistener.eft_rate import abelian_higgs_3d
        self.g = g
        self.rm = EP.RateModel.__new__(EP.RateModel)
        def load(kind):
            """the action cache of one kind, with the low-temperature extension of
            compute_actions.py --low prepended where it exists"""
            from compute_actions import cache_tag
            tag = cache_tag(g)
            d = json.load(open(os.path.join(paths.ACTION_CACHE, f"act_{kind}_g{tag}.json")))
            assert abs(d["g"] - g) < 1e-9, f"the {kind} cache for g = {g} was computed at g = {d['g']}"
            low = os.path.join(paths.ACTION_CACHE, f"act_{kind}_g{tag}_low.json")
            if os.path.exists(low):
                d["rows"] = sorted(json.load(open(low))["rows"] + d["rows"], key=lambda r: r["T"])
            # the extension above the grid (compute_actions.py --high), for every coupling
            high = os.path.join(paths.ACTION_CACHE, f"act_{kind}_g{tag}_high.json")
            if os.path.exists(high):
                d["rows"] = sorted(d["rows"] + json.load(open(high))["rows"], key=lambda r: r["T"])
            # the further extension (compute_actions.py --higher) for the slow couplings at small v
            for suf in ("_high2", "_high3"):
                ext = os.path.join(paths.ACTION_CACHE, f"act_{kind}_g{tag}{suf}.json")
                if os.path.exists(ext):
                    d["rows"] = sorted(d["rows"] + json.load(open(ext))["rows"], key=lambda r: r["T"])
            return d
        a1 = load("nlo1")
        self.anchor = a1["anchor"]
        self.T = np.array([r["T"] for r in a1["rows"]])
        self.S3, self.D6 = {}, {}
        for k in ("nlo1", "nlo05", "nlo2", "lom", "mu3x05", "mu3x2"):
            rows = load(k)["rows"]
            # the kinds are paired row by row: every cache must lie on the temperatures of nlo1 (atol = 0: with
            # numpy's default atol = 1e-8 GeV the check would pass any grid below 1e-8 GeV)
            Tk = np.array([r["T"] for r in rows])
            assert len(Tk) == len(self.T) and np.allclose(Tk, self.T, rtol=1e-9, atol=0.0), \
                f"g = {g}: the {k} actions lie on other temperatures than nlo1"
            self.S3[k] = np.array([r["S3"] for r in rows])
            # Delta S_dim6 of this kind on its own bounce; zero with EXACT_DIM6=0
            if DIM6:
                missing = [r["T"] for r in rows if np.isfinite(r["S3"]) and "dSkin" not in r]
                if missing:
                    raise RuntimeError(f"g = {g}: the {k} actions have no dimension-six shifts at {len(missing)} "
                                       "temperatures; run compute_actions.py --add-dim6 (or set EXACT_DIM6=0)")
                self.D6[k] = np.array([r["dS6"] + r["dSkin"] if np.isfinite(r["S3"]) else 0.0 for r in rows])
            else:
                self.D6[k] = np.zeros(len(rows))
        self.x3 = np.array([r.get("x3", np.nan) for r in a1["rows"]])
        import rate_history_u1 as R
        from compute_actions import line_point, scan_anchors
        gt = g if g >= G_TABLE_MIN - 1e-9 else G_TABLE_MIN

        def tables(pdir):
            corr = {}
            for m in ("1", "0.5", "2"):
                t = json.load(open(os.path.join(pdir, f"table_mu{m}.json")))["table"]
                if t.get("S3_LO") is None:
                    raise RuntimeError(f"{pdir}: the Delta S table has no S3_LO, needed for the ratio continuation")
                corr[m] = _RatioCorr(t["T_GeV"], t["delta_S"], t["S3_LO"])
            return corr
        try:
            if blend is not None:
                raise FileNotFoundError("blend requested")
            pdir = R.point_dir(gt)
            self.corr = tables(pdir)
        except FileNotFoundError:
            # g >= G_TABLE_MIN without a line-scan point (or blend=(ga, gb) for the leave-one-out check):
            # Delta S interpolated between the neighbouring couplings that have tables
            A = scan_anchors()
            ga, gb = blend or (max(x for x in A if x < g), min(x for x in A if x > g))
            w = (g - ga) / (gb - ga)
            anchor_g = line_point(g)[1] if blend is None else json.load(open(os.path.join(
                R.point_dir(g), "summary.json")))["anchor_GeV"]
            ca, cb = tables(R.point_dir(ga)), tables(R.point_dir(gb))
            sa, sb = A[ga][1] / anchor_g, A[gb][1] / anchor_g
            self.corr = {m: _Blend(ca[m], cb[m], sa, sb, w) for m in ca}
        # the scalar fraction of the mu3 logarithm of m3^2 (see exponent())
        f = []
        for T in self.T:
            q = R.dr.matched_soft_parameters(g, 0.0, R.MU0, T, mu4_over_T=R.MU4, matching_order="NLO", mu3=g * T)
            g3, l3, lvl = q["g3sq"], q["lam3"], q["lambda_vl"]
            sc = -16 * g3 * l3 + 16 * l3**2
            f.append(sc / (8 * g3**2 + sc + lvl**2))
        self.f_scalar = np.array(f)
        self.T_scale = 1.0
        if g < G_TABLE_MIN - 1e-9:     # table of the lowest coupling in units of the percolation temperature
            a_min = json.load(open(os.path.join(pdir, "summary.json")))["anchor_GeV"]
            self.T_scale = a_min / line_point(g)[1]
        self.v_line = line_point(g)[0]
        self.dV_line = R.vacuum_energy(g, self.v_line)
        self.table_T = (min(self.corr["1"].T_range), max(self.corr["1"].T_range))
        self.corr_tab, self.T_scale_tab = self.corr, self.T_scale
        if DS_SOURCE == "grid":
            self.corr = {m: _GridCorr(g, f, R.MU4) for m, f in MU4_FACTORS.items()}
            self.T_scale = 1.0

    def dS(self, m, S3, tables=False):
        corr, scale = (self.corr_tab, self.T_scale_tab) if tables else (self.corr, self.T_scale)
        return np.array([float(corr[m](T * scale, S3_LO=s)) if np.isfinite(s) and s > 0 else 0.0
                         for T, s in zip(self.T, S3)])

    def in_range(self, T_line):
        """Whether Delta S at the line temperature T_line comes from inside its source: S_LO inside the grid's range,
        or T inside the table."""
        if DS_SOURCE == "tables":
            return bool(self.table_T[0] <= T_line * self.T_scale <= self.table_T[1])
        fin = np.isfinite(self.S3["nlo1"])
        S = float(np.interp(math.log(T_line), np.log(self.T[fin]), self.S3["nlo1"][fin]))
        lo, hi = self.corr["1"].S_range
        return bool(lo <= S <= hi)

    def exponent(self, variant):
        S3 = self.S3["nlo1"]; dS = self.dS("1", S3); D6 = self.D6["nlo1"]
        nlo4 = S3 + dS                       # the dimension-four exponent
        nlo = with_S6(S3, dS, D6)
        if variant in ("nlo", "perc1e", "dark"):
            return nlo
        if variant == "dim4":
            return nlo4
        if variant in ("mu3x05", "mu3x2"):
            # The EFT scale mu3 enters m3^2 through ln(mu3) (8 g3^4 - 16 g3^2 lam3 + 16 lam3^2 + lVL^2)/32 pi^2;
            # to first order in the bounce the shift of S3 is linear in ln mu3, and the two-loop soft
            # potential inside Delta S cancels its gauge and A0 part (-(g3^4/8 pi^2 + lVL^2/64 pi^2) phi^2
            # ln mu3), not the scalar part. The consistent variation is therefore the scalar fraction f
            # of the linear shift plus the second-order remainder of the LO bounce.
            Sp, Sm = self.S3["mu3x2"], self.S3["mu3x05"]
            lin = 0.5 * (Sp - Sm)
            second = 0.5 * (Sp + Sm - 2 * S3)
            sgn = 1.0 if variant == "mu3x2" else -1.0
            return nlo + sgn * self.f_scalar * lin + second
        if variant == "dStab":
            return with_S6(S3, self.dS("1", S3, tables=True), D6)
        if variant == "mu05":
            return with_S6(self.S3["nlo05"], self.dS("0.5", self.S3["nlo05"]), self.D6["nlo05"])
        if variant == "mu2":
            return with_S6(self.S3["nlo2"], self.dS("2", self.S3["nlo2"]), self.D6["nlo2"])
        nxt = self.x3 * np.maximum(np.abs(dS), 1.0)
        if variant in ("hord+", "hord-"):
            # x3 |S_NLO - S_LO|, capped at x3 S_NLO: near the critical temperature of the LO-matched theory
            # S_LO diverges (and above it is NaN), so that the uncapped step can exceed the action itself, and
            # hord+ would nucleate spuriously at 4 to 22 times T_p. The cap changes only temperatures far above T_p.
            # the step of the loop expansion, without the dimension-six shifts (a separate expansion)
            step = self.x3 * np.fmin(np.abs(nlo4 - self.S3["lom"]), nlo4)
            return nlo - step if variant == "hord+" else nlo + step
        if variant in ("latc+", "latc-"):
            shift = 0.2 * float(np.interp(math.log(self.anchor), np.log(self.T), nlo))
            return nlo - shift if variant == "latc+" else nlo + shift
        if variant in ("latp+", "latp-"):
            return nlo * (0.8 if variant == "latp+" else 1.2)
        if variant == "next+":
            return nlo - nxt
        if variant == "next-":
            return nlo + nxt
        if variant == "adynT":
            return nlo - 2.36
        if variant == "pref+":
            return nlo - math.log(10)
        if variant == "pref-":
            return nlo + math.log(10)
        if variant == "lorate":
            return S3 + D6
        if variant == "fulllo":
            return self.S3["lom"] + self.D6["lom"]
        raise ValueError(variant)


def run_g(g):
    import rate_history_u1 as R
    m = Model(g)
    rm = m.rm
    rm.g, rm.v_line, rm.dV_line, rm.Ts, rm.corr = g, m.v_line, m.dV_line, m.T, None
    out = {k: np.full((len(VARIANTS), len(LOGV)), np.nan) for k in KEYS}
    treh = np.full((2, len(LOGV)), np.nan)
    for iv, lv in enumerate(LOGV):
        v = 10.0 ** lv
        dV = m.dV_line * (v / m.v_line) ** 4
        for j, dof in enumerate((0.0, 4.0)):
            gr = lambda T: R.g_rho(T) - R.DARK_DOF + dof
            treh[j, iv] = brentq(lambda T: math.pi**2 / 30 * gr(T) * T**4 - dV, 1e-6, 1e4)
        for k, var in enumerate(VARIANTS):
            S = m.exponent(var)
            sample(rm, m.T, S)
            r = EP.percolate(rm, v=v, time="dark" if var == "dark" else "entropy",
                             f_perc=(1 - math.exp(-1)) if var == "perc1e" else EP.F_PERC)
            out["T_min_window"][k, iv] = float(np.min(m.T[np.isfinite(S)])) * v / m.v_line
            if r["ok"] and r["T_p"] < 1.5 * out["T_min_window"][k, iv]:
                # percolation at the lowest temperatures of the window: the slopes there are
                # one-sided and the result depends on where the window ends
                r = dict(ok=False, rising=True)
            if r["ok"]:
                # the hottest temperature with a physical exponent: near T_c the next-order construction
                # (hord+) can turn negative, where the rate is not defined and does not count
                fin = np.isfinite(S) & (S > 0)
                if float(S[fin][np.argmax(m.T[fin])]) - r["S_p"] < HOT_MARGIN:
                    # percolation at the hottest temperatures of the window: the rate above it is not
                    # negligible, and T_p is set by where the grid starts
                    r = dict(ok=False, rising=True)
            if not r["ok"]:
                out["window_limited"][k, iv] = float(r["rising"])
                # the largest true-vacuum fraction the window reaches, so that the edge of
                # percolation (P_t_end = f_perc) can be interpolated between the nodes
                out["P_t_end"][k, iv] = float(r.get("P_t_end", np.nan))
                continue
            out["window_limited"][k, iv] = 0.0
            # completion inside the window, or undecided because the rate still rises at its end
            out["f_window_limited"][k, iv] = float(not np.isfinite(r["T_f"]) and r["f_rising"])
            for key in KEYS[:-5]:
                out[key][k, iv] = r[key]
            out["in_table"][k, iv] = float(m.in_range(r["T_p"] / (v / m.v_line)))
            # (beta/H)_S at T_nuc
            s = v / m.v_line
            fin = np.isfinite(S)
            lnT = np.log(m.T[fin] * s); dS_dlnT = np.gradient(S[fin], lnT)
            if np.isfinite(r["T_nuc"]):
                out["betaH_S_nuc"][k, iv] = float(np.interp(math.log(r["T_nuc"]), lnT, dS_dlnT))
    return g, out, treh


if __name__ == "__main__":
    t0 = time.time()
    with Pool(len(G)) as pool:
        res = pool.map(run_g, G)
    res.sort(key=lambda x: x[0])
    tab = {key: np.array([r[1][key] for r in res]) for key in KEYS}      # (g, variant, logv)
    np.savez(os.environ.get("EXACT_TABLE_OUT", paths.work("exact_table.npz")), G=np.array(G), LOGV=LOGV, VARIANTS=np.array(VARIANTS),
             T_reh_SM=np.array([r[2][0] for r in res]), T_reh_SM4=np.array([r[2][1] for r in res]), **tab)
    k = VARIANTS.index("nlo") if "nlo" in VARIANTS else 0; j = int(np.argmin(np.abs(LOGV - 0.25)))
    for (g, o, tr) in res:
        print(f"g={g:.3f} v=10^{LOGV[j]}: T_p {o['T_p'][k, j]:.3e} (b/H)_R {o['betaH_R'][k, j]:.2f} (b/H)_S {o['betaH_S'][k, j]:.2f} "
              f"dlnV {o['dlnV_dt_over_H'][k, j]:+.2f}; (b/H)_R vs v: " + " ".join(f"{x:.1f}" for x in o["betaH_R"][k]), flush=True)
    print(f"wrote exact_table.npz, {time.time() - t0:.0f} s")
