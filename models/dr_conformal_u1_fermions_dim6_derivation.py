"""One-loop derivation of the dimension-six coefficients of the conformal U(1) with fermions soft 3D EFT.

Derives, from the 4D theory of `TL_conformal_u1_fermions.py`, the coefficients of

    c6    (Phi^+Phi)^3                         (dimensionless in 3D)
    c_kin (Phi^+Phi)(D_i Phi)^+(D_i Phi)       (mass dimension -1 in 3D)

in the soft 3D EFT, in the normalisation of arXiv:2610.12284 app. A (3D field Phi, L3 contains
+c6 (Phi^+Phi)^3 + c_kin (Phi^+Phi)(D Phi)^+(D Phi)). The bosonic parts must come out as
`c6_bosonic` and `cD2phi4_bosonic` of `dr_conformal_u1_fermions_matching.py`; that is the check of the
method. The same code with the fermions switched on gives their contribution.

Method
------
1. Hard one-loop effective action in the background field method: Phi -> Phi^ + eta, A -> A^ + a,
   background R_xi gauge G = d.a - i xi g (eta^+ Phi^ - Phi^+ eta), so that the effective action is
   invariant under background gauge transformations. Gauge fields, eta and the ghosts run over the
   non-zero bosonic Matsubara modes, the fermions over all fermionic ones. The two Weyl fermions
   chi_1 (Q = -1/2, Yukawa (y/2) Phi chi chi) and chi_2 (Q = +1/2, (y/2) Phi^* chi chi) are written
   as Majorana spinors with Euclidean Dirac operator
       gamma_mu (d_mu + i g Q gamma_5 A_mu) + m P_L + m^* P_R,   m = y Phi (y Phi^* for chi_2),
   which is covariant under Phi -> e^{i g theta} Phi; each contributes -(1/2) Tr ln.
2. Every interaction is an insertion (all loop propagators massless), so the coefficient of a
   product of background plane waves is a sum of one-loop diagrams. The integrand is expanded in
   the external (spatial) momenta to the order of a dimension-six operator; every resulting
   sum-integral is finite and proportional to zeta(3)/T^2, so d = 3 and D = 4 are exact.
3. The result is matched, plane wave by plane wave, onto the off-shell (Green's) basis
       O1 = rho^3,  O2 = rho |D Phi|^2,  O4 = rho (Phi^+ D^2 Phi + h.c.),
       O5 = i rho (Phi^+ D^2 Phi - h.c.),  O6 = |D^2 Phi|^2,  O7 = (d_i F_ij)^2,  O8 = d_i F_ij J_j
   with rho = Phi^+Phi and J_j = 2 Im(Phi^+ D_j Phi). O3 = (d rho)^2 is not in the basis: by parts
   it is -2 O2 - O4. The system is overdetermined; that it has a unique solution checks background
   gauge invariance and the completeness of the basis.
4. On-shell reduction with the tree-level 3D equations of motion, D^2 Phi = 2 lam3 rho Phi (+ m3^2,
   B0 terms) and d_i F_ij = -g3 J_j:  O4 -> 4 lam3 rho^3,  O5 -> 0,  O6 -> 4 lam3^2 rho^3,
   O7 -> g3^2 J^2,  O8 -> -g3 J^2,  and J^2 = 4 O2 - O3 = 6 O2 + O4.

The bosonic part reproduces arXiv:2610.12284 app. A exactly and is independent of xi, although the
Green's-basis coefficients are not; both are checked in tests/test_dim6_fermions.py.

Sympy is needed only here, not by the pipeline: `dr_conformal_u1_fermions_matching.py` carries the
resulting closed forms, and tests/test_dim6_fermions.py checks them against this derivation.
"""
from __future__ import annotations

import itertools
from functools import lru_cache

import sympy as sp
from sympy.polys.domains import QQ_I
from sympy.polys.rings import ring

I = sp.I

# loop momentum P = (w, px, py, pz), iD = 1/P^2, tau counts external momenta
R, w, px, py, pz, iD, tau, g, lam, y, xi = ring("w,px,py,pz,iD,tau,g,lam,y,xi", QQ_I)
_GENS = dict(w=w, px=px, py=py, pz=pz, iD=iD, tau=tau, g=g, lam=lam, y=y, xi=xi)
_TAU = R.gens.index(tau)
_SYM = {name: sp.Symbol(name) for name in _GENS}
GS, LAMS, YS, XIS, TAU_S = _SYM["g"], _SYM["lam"], _SYM["y"], _SYM["xi"], _SYM["tau"]


# ----------------------------------------------------------------------
# plane-wave algebra: a field is {momentum (4-tuple): coefficient}
# ----------------------------------------------------------------------
class PW(dict):
    def __add__(self, other):
        out = PW(self)
        for k, c in other.items():
            out[k] = out.get(k, 0) + c
        return out

    def __neg__(self):
        return PW({k: -c for k, c in self.items()})

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if not isinstance(other, PW):
            return PW({k: c * other for k, c in self.items()})
        out = PW()
        for k1, c1 in self.items():
            for k2, c2 in other.items():
                k = tuple(sp.expand(a + b) for a, b in zip(k1, k2))
                out[k] = out.get(k, 0) + c1 * c2
        return out

    __rmul__ = __mul__

    def total(self):
        return sp.expand(sum(self.values()))


def d(f: PW, mu: int) -> PW:
    """Derivative: d_mu e^{ikx} = i k_mu e^{ikx}."""
    return PW({k: I * k[mu] * c for k, c in f.items()})


def plane(amp, mom) -> PW:
    return PW({tuple(sp.sympify(m) for m in mom): amp})


ZERO4 = (0, 0, 0, 0)


class Background:
    """Phi^ = sum_i u_i e^{i k_i x}, Phi^+ = sum_j ub_j e^{i kb_j x}, A^_mu = sum_l eps_l e^{i q_l x}.

    Phi^ and Phi^+ carry independent amplitudes and momenta, as in an off-shell n-point function.
    Momenta are spatial 3-vectors; A^_0 = 0 (static spatial background).
    """

    def __init__(self, phis=(), phibars=(), gauges=()):
        self.amps = []
        self.mom = {}
        self.Phi, self.Phid = PW(), PW()
        self.A = [PW() for _ in range(4)]
        for i, k in enumerate(phis):
            a = sp.Symbol(f"u{i}")
            self._add(a, k)
            self.Phi = self.Phi + plane(a, self._scaled(k))
        for i, k in enumerate(phibars):
            a = sp.Symbol(f"ub{i}")
            self._add(a, k)
            self.Phid = self.Phid + plane(a, self._scaled(k))
        for i, (eps, k) in enumerate(gauges):
            a = sp.Symbol(f"e{i}")
            self._add(a, k)
            for mu in range(1, 4):
                self.A[mu] = self.A[mu] + plane(a * sp.sympify(eps[mu - 1]), self._scaled(k))

    @staticmethod
    def _scaled(k):
        """Background momenta carry tau, so that derivative vertices count in the expansion too."""
        return (0,) + tuple(TAU_S * x for x in k)

    def _add(self, a, k):
        self.amps.append(a)
        self.mom[a] = tuple(sp.sympify(x) for x in k)

    def kmom(self, subset):
        return tuple(sum((self.mom[a][i] for a in subset), sp.S(0)) for i in range(3))


# ----------------------------------------------------------------------
# the bosonic fluctuation operator
# ----------------------------------------------------------------------
P1 = sp.symbols("P1_0:4")
P2 = sp.symbols("P2_0:4")
BOSONS = ("eta1", "eta2", "a0", "a1", "a2", "a3")


def _bosonic_lagrangian(bg: Background):
    """L of the 4D theory with Phi = Phi^ + eta, A = A^ + a, in background R_xi gauge.

    The fluctuations are sigma_X e^{i P1 x} + tau_X e^{i P2 x}; the coefficient of sigma_X tau_Y in
    L is the kernel K_XY(P1, P2) of S2 = (1/2) phi K phi.
    """
    sig = {X: sp.Symbol(f"sig_{X}") for X in BOSONS}
    tau_ = {X: sp.Symbol(f"tau_{X}") for X in BOSONS}
    fl = {X: plane(sig[X], P1) + plane(tau_[X], P2) for X in BOSONS}
    # eta = eta1 + i eta2 without the 1/sqrt 2, which keeps every vertex Gauss-rational; the
    # eta block of the free operator is then 2 P^2 (see _boson_propagator)
    eta = fl["eta1"] + fl["eta2"] * I
    etad = fl["eta1"] - fl["eta2"] * I
    Phi, Phid = bg.Phi + eta, bg.Phid + etad
    A = [bg.A[mu] + fl[f"a{mu}"] for mu in range(4)]
    a = [fl[f"a{mu}"] for mu in range(4)]

    L = PW()
    for mu in range(4):
        DPhi = d(Phi, mu) - A[mu] * Phi * (I * GS)
        DPhid = d(Phid, mu) + A[mu] * Phid * (I * GS)
        L = L + DPhid * DPhi
    rho = Phid * Phi
    L = L + rho * rho * LAMS
    # (1/4) F^2 = (1/2)(d_mu A_nu)^2 - (1/2)(d.A)^2 up to a total derivative
    for mu in range(4):
        for nu in range(4):
            L = L + d(A[nu], mu) * d(A[nu], mu) * sp.Rational(1, 2)
    divA = PW()
    for mu in range(4):
        divA = divA + d(A[mu], mu)
    L = L - divA * divA * sp.Rational(1, 2)
    # gauge fixing (1/2 xi) G^2, G = d.a - i xi g (eta^+ Phi^ - Phi^+ eta)
    diva = PW()
    for mu in range(4):
        diva = diva + d(a[mu], mu)
    G = diva - (etad * bg.Phi - bg.Phid * eta) * (I * XIS * GS)
    L = L + G * G * (1 / (2 * XIS))
    return sp.expand(L.total()), sig, tau_


@lru_cache(maxsize=None)
def _bosonic_vertices(bg_key):
    bg = _BG_REGISTRY[bg_key]
    Ltot, sig, tau_ = _bosonic_lagrangian(bg)
    poly = sp.Poly(Ltot, *[sig[X] for X in BOSONS], *[tau_[X] for X in BOSONS], *bg.amps)
    nb = len(BOSONS)
    na = len(bg.amps)
    verts = {}
    for monom, coeff in poly.terms():
        ms, mt, ma = monom[:nb], monom[nb:2 * nb], monom[2 * nb:]
        if sum(ms) != 1 or sum(mt) != 1 or any(e > 1 for e in ma):
            continue
        A_, B_ = ms.index(1), mt.index(1)
        S = frozenset(bg.amps[i] for i in range(na) if ma[i])
        verts.setdefault(S, sp.zeros(nb, nb))
        verts[S][A_, B_] += coeff
    return verts


_BG_REGISTRY = {}


def _register(bg):
    key = len(_BG_REGISTRY)
    _BG_REGISTRY[key] = bg
    return key


# ----------------------------------------------------------------------
# series in tau (ring elements, truncated)
# ----------------------------------------------------------------------
def _trunc(p, N):
    return R({m: c for m, c in p.items() if m[_TAU] <= N})


def _mat_mul(Am, Bm, N):
    n, k, m = len(Am), len(Bm), len(Bm[0])
    out = [[R.zero] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            acc = R.zero
            for l in range(k):
                if Am[i][l] and Bm[l][j]:
                    acc += Am[i][l] * Bm[l][j]
            out[i][j] = _trunc(acc, N)
    return out


def _ploop(K):
    """(P - tau K) with P = (w, p) and K a spatial 3-vector: ring 4-vector."""
    return (w, px - tau * R(K[0]), py - tau * R(K[1]), pz - tau * R(K[2]))


def _inv_sq(K, N):
    """1/(P - tau K)^2 = iD sum_j (-delta iD)^j, delta = -2 tau p.K + tau^2 K^2."""
    delta = -2 * tau * (px * R(K[0]) + py * R(K[1]) + pz * R(K[2])) + tau**2 * R(sum(k * k for k in K))
    out, term = R.zero, iD
    for _ in range(N + 1):
        out += term
        term = _trunc(-term * delta * iD, N)
    return _trunc(out, N)


def _sym_to_ring(expr):
    return R.from_expr(sp.expand(expr)) if expr != 0 else R.zero


def _subs_vertex(M, Pout, Pin, N):
    """Vertex V(P_out, P_in) = K(-P_out, P_in) as a matrix of ring elements."""
    out = []
    for i in range(M.rows):
        row = []
        for j in range(M.cols):
            e = sp.expand(M[i, j])
            if e == 0:
                row.append(R.zero)
                continue
            p = sp.Poly(e, *P1, *P2)
            acc = R.zero
            for monom, c in p.terms():
                t = _sym_to_ring(c)
                for mu in range(4):
                    if monom[mu]:
                        t *= (-Pout[mu]) ** monom[mu]
                    if monom[4 + mu]:
                        t *= Pin[mu] ** monom[4 + mu]
                acc += t
            row.append(_trunc(acc, N))
        out.append(row)
    return out


def _boson_propagator(K, N):
    """Free propagator at P - tau K: eta block 1/(2 Q^2), gauge block (delta - (1 - xi) Q Q / Q^2)/Q^2."""
    s = _inv_sq(K, N)
    Q = _ploop(K)
    s2 = _trunc(s * s, N)
    G = [[R.zero] * 6 for _ in range(6)]
    G[0][0] = G[1][1] = _trunc(s * R(sp.Rational(1, 2)), N)
    for mu in range(4):
        for nu in range(4):
            v = (s if mu == nu else R.zero) - _trunc((1 - xi) * Q[mu] * Q[nu] * s2, N)
            G[2 + mu][2 + nu] = _trunc(v, N)
    return G


# ----------------------------------------------------------------------
# sum over ordered partitions = expansion of Tr ln
# ----------------------------------------------------------------------
def _loop_sum(bg, vertex_fn, propagator_fn, N):
    """sum_n (-1)^{n+1}/n sum_{ordered partitions} tr[G V G V ...], coefficient of tau^N.

    The trace is cyclic and a constant shift of the loop momentum leaves every (finite or
    dimensionally regulated) term unchanged, so the n cyclic orderings of a partition are equal:
    only orderings whose first block holds the first amplitude are summed, without the 1/n.
    Partial products are shared between orderings with a common beginning.
    """
    amps = list(bg.amps)
    first = amps[0]
    total = [R.zero]

    def extend(prod, remaining, K, n):
        if not remaining:
            tr = R.zero
            for i in range(len(prod)):
                tr += prod[i][i]
            total[0] += _trunc(tr * R((-1) ** (n + 1)), N)
            return
        rem = sorted(remaining, key=str)
        must = [first] if n == 0 else []
        pool = [a for a in rem if a not in must]
        for r in range(0, len(pool) + 1):
            for extra in itertools.combinations(pool, r):
                S = frozenset(must) | frozenset(extra)
                if not S or vertex_fn(S, None, None, N, probe=True) is None:
                    continue
                k = bg.kmom(S)
                K2 = tuple(a + b for a, b in zip(K, k))
                GV = _mat_mul(propagator_fn(K, N), vertex_fn(S, K, K2, N), N)
                extend(GV if prod is None else _mat_mul(prod, GV, N), remaining - S, K2, n + 1)

    extend(None, frozenset(amps), (0, 0, 0), 0)
    return R({m: c for m, c in total[0].items() if m[_TAU] == N})


# ----------------------------------------------------------------------
# sum-integrals
# ----------------------------------------------------------------------
def _dfact(n):
    return sp.S(1) if n <= 0 else sp.factorial2(n)


@lru_cache(maxsize=None)
def _radial(s, n):
    """int d^3p/(2pi)^3 p^{2s}/(p^2 + w^2)^n = C |w|^{3+2s-2n}; returns C pi/4, so that
    T sum_{w != 0, bosonic} gives C pi/4 * zeta3/(pi^4 T^2)."""
    h = sp.Rational(3, 2)
    C = sp.gamma(s + h) * sp.gamma(n - s - h) / ((4 * sp.pi) ** h * sp.gamma(h) * sp.gamma(n))
    return sp.nsimplify(sp.simplify(C * sp.pi / 4))


def integrate(poly, fermionic=False):
    """Sum-integral of a ring polynomial in (w, p, iD); result in units of zeta3/(pi^4 T^2), as a
    sympy expression in g, lam, y, xi. Every monomial must be of total dimension -3 in |w|."""
    out = sp.S(0)
    gi = [R.gens.index(x) for x in (w, px, py, pz, iD)]
    for m, c in poly.items():
        e, a, b, cc, n = (m[i] for i in gi)
        if e % 2 or a % 2 or b % 2 or cc % 2:
            continue
        s2 = a + b + cc
        ang = _dfact(a - 1) * _dfact(b - 1) * _dfact(cc - 1) / _dfact(s2 + 1)
        s = s2 // 2
        assert e + 3 + 2 * s - 2 * n == -3, (e, s, n)
        val = ang * _radial(s, n)
        coup = sp.S(1)
        for name in ("g", "lam", "y", "xi"):
            k = m[R.gens.index(_GENS[name])]
            coup *= _SYM[name] ** k
        cs = R.domain.to_sympy(c)
        out += cs * val * coup
    if fermionic:
        out *= 7
    return sp.expand(out)


# ----------------------------------------------------------------------
# the three species
# ----------------------------------------------------------------------
def bosonic_loop(bg: Background, N: int):
    """(1/2) Tr ln of the gauge-scalar fluctuation operator."""
    key = _register(bg)
    verts = _bosonic_vertices(key)

    def vertex_fn(S, Kout, Kin, N_, probe=False):
        if not S or S not in verts:
            return None
        if probe:
            return True
        return _subs_vertex(verts[S], _ploop(Kout), _ploop(Kin), N_)

    return integrate(_loop_sum(bg, vertex_fn, _boson_propagator, N)) / 2


def ghost_loop(bg: Background, N: int):
    """-Tr ln(-d^2 + 2 xi g^2 rho^) of the (uncharged, complex) ghost."""
    rho = bg.Phid * bg.Phi
    poly = sp.Poly(rho.total(), *bg.amps)
    verts = {}
    for monom, c in poly.terms():
        if any(e > 1 for e in monom):
            continue
        verts[frozenset(a for a, e in zip(bg.amps, monom) if e)] = 2 * XIS * GS**2 * c

    def vertex_fn(S, Kout, Kin, N_, probe=False):
        if S not in verts:
            return None
        if probe:
            return True
        return [[_sym_to_ring(verts[S])]]

    def prop(K, N_):
        return [[_inv_sq(K, N_)]]

    return -integrate(_loop_sum(bg, vertex_fn, prop, N))


# Euclidean gamma matrices, chiral representation
_s = [sp.Matrix([[0, 1], [1, 0]]), sp.Matrix([[0, -I], [I, 0]]), sp.Matrix([[1, 0], [0, -1]])]
_E2 = sp.eye(2)
_A = [_E2] + [-I * sk for sk in _s]


def _gam(Amat):
    Z = sp.zeros(2, 2)
    return sp.Matrix(sp.BlockMatrix([[Z, Amat], [Amat.H, Z]]))


GAMMA = [_gam(Am) for Am in _A]
GAMMA5 = GAMMA[0] * GAMMA[1] * GAMMA[2] * GAMMA[3]
PL = (sp.eye(4) - GAMMA5) / 2
PR = (sp.eye(4) + GAMMA5) / 2


def fermion_loop(bg: Background, N: int, charge, conj_phi: bool):
    """-(1/2) Tr ln of gamma_mu (d_mu + i g Q gamma5 A_mu) + m P_L + m^* P_R, m = y Phi (or y Phi^*).

    The fermionic Matsubara sums run over all modes.
    """
    # m P_L + m^* P_R: Phi^ enters m (chi_1) or m^* (chi_2), Phi^+ the other one
    verts = {}
    for a in bg.amps:
        name = str(a)
        if name.startswith("ub"):
            verts[frozenset([a])] = (PL if conj_phi else PR) * YS
        elif name.startswith("u"):
            verts[frozenset([a])] = (PR if conj_phi else PL) * YS
        else:   # gauge vertex i g Q gamma_mu gamma5 eps_mu
            M = sp.zeros(4, 4)
            for mu in range(1, 4):
                M += I * GS * charge * GAMMA[mu] * GAMMA5 * bg.A[mu].total().coeff(a)
            verts[frozenset([a])] = M

    def vertex_fn(S, Kout, Kin, N_, probe=False):
        if S not in verts:
            return None
        if probe:
            return True
        M = verts[S]
        return [[_sym_to_ring(M[i, j]) for j in range(4)] for i in range(4)]

    def prop(K, N_):
        # (i Qslash)^{-1} = -i Qslash / Q^2
        s = _inv_sq(K, N_)
        Q = _ploop(K)
        out = [[R.zero] * 4 for _ in range(4)]
        for mu in range(4):
            for i in range(4):
                for j in range(4):
                    gij = GAMMA[mu][i, j]
                    if gij != 0:
                        out[i][j] += _trunc(_sym_to_ring(-I * gij) * Q[mu] * s, N_)
        return out

    return -integrate(_loop_sum(bg, vertex_fn, prop, N), fermionic=True) / 2


def fermions_loop(bg: Background, N: int):
    """chi_1 (Q = -1/2, couples to Phi) and chi_2 (Q = +1/2, couples to Phi^*)."""
    return (fermion_loop(bg, N, sp.Rational(-1, 2), conj_phi=False)
            + fermion_loop(bg, N, sp.Rational(1, 2), conj_phi=True))


# ----------------------------------------------------------------------
# operator side
# ----------------------------------------------------------------------
def _ops(bg: Background):
    """Multilinear coefficient of every Green's-basis operator on the background (spatial D only)."""
    Phi, Phid, A = bg.Phi, bg.Phid, bg.A
    DPhi = [d(Phi, i) - A[i] * Phi * (I * GS) for i in range(4)]
    DPhid = [d(Phid, i) + A[i] * Phid * (I * GS) for i in range(4)]

    def Dcov(f, i):
        return d(f, i) - A[i] * f * (I * GS)

    def Dcovd(f, i):
        return d(f, i) + A[i] * f * (I * GS)

    sp_idx = range(1, 4)
    rho = Phid * Phi
    D2Phi = PW()
    D2Phid = PW()
    for i in sp_idx:
        D2Phi = D2Phi + Dcov(DPhi[i], i)
        D2Phid = D2Phid + Dcovd(DPhid[i], i)
    kin = PW()
    for i in sp_idx:
        kin = kin + DPhid[i] * DPhi[i]
    F = [[d(A[j], i) - d(A[i], j) for j in range(4)] for i in range(4)]
    dF = [PW() for _ in range(4)]
    for j in sp_idx:
        for i in sp_idx:
            dF[j] = dF[j] + d(F[i][j], i)
    J = [(Phid * DPhi[j] - DPhid[j] * Phi) * (-I) for j in range(4)]
    ops = {
        "O1": rho * rho * rho,
        "O2": rho * kin,
        "O4": rho * (Phid * D2Phi + D2Phid * Phi),
        "O5": rho * (Phid * D2Phi - D2Phid * Phi) * I,
        "O6": D2Phid * D2Phi,
        "O7": sum((dF[j] * dF[j] for j in sp_idx), PW()),
        "O8": sum((dF[j] * J[j] for j in sp_idx), PW()),
    }
    out = {}
    for name, op in ops.items():
        p = sp.Poly(op.total(), *bg.amps)
        target = tuple(1 for _ in bg.amps)
        out[name] = sp.expand(p.coeff_monomial(target).subs(TAU_S, 1))
    return out


# ----------------------------------------------------------------------
# matching
# ----------------------------------------------------------------------
_RNG_MOMENTA = [
    ((1, 0, 2), (-2, 1, 0), (0, 3, -1)),
    ((2, -1, 1), (1, 2, -3), (-1, 0, 1)),
    ((0, 1, -1), (3, 1, 2), (-2, -2, 1)),
    ((1, 1, 1), (-1, 2, 0), (2, -3, 1)),
    ((3, 0, -1), (0, -1, 2), (1, 2, 2)),
    ((-1, 2, 2), (2, 0, 1), (0, -1, -3)),
]


def _configs(kind):
    """Backgrounds (with momentum conservation) and the order in tau that is dimension six."""
    out = []
    if kind == "6pt":
        out.append((Background(phis=[(0, 0, 0)] * 3, phibars=[(0, 0, 0)] * 3), 0))
    elif kind == "4pt":
        for k1, k2, k3 in _RNG_MOMENTA:
            k4 = tuple(-(a + b + c) for a, b, c in zip(k1, k2, k3))
            out.append((Background(phis=[k1, k2], phibars=[k3, k4]), 2))
    elif kind == "2pt":
        out.append((Background(phis=[(1, 2, -1)], phibars=[(-1, -2, 1)]), 4))
    elif kind == "AA":
        out.append((Background(gauges=[((1, 0, 2), (1, 2, -1)), ((2, 1, 0), (-1, -2, 1))]), 4))
        out.append((Background(gauges=[((1, -1, 0), (0, 1, 3)), ((0, 2, 1), (0, -1, -3))]), 4))
    elif kind == "3pt":
        for k1, k2, eps in _RNG_MOMENTA[:4]:
            k3 = tuple(-(a + b) for a, b in zip(k1, k2))
            out.append((Background(phis=[k1], phibars=[k2], gauges=[(eps, k3)]), 3))
    return out


def green_coefficients(include_bosons=True, include_fermions=True, include_ghosts=None):
    """Green's-basis coefficients c1..c8 in units of zeta3/(pi^4 T^2) times the power of T fixed by
    dimension (4D normalisation, T = 1). Solves the overdetermined system and checks consistency."""
    if include_ghosts is None:
        include_ghosts = include_bosons
    unknowns = [sp.Symbol(f"c{i}") for i in (1, 2, 4, 5, 6, 7, 8)]
    eqs = []
    for kind in ("6pt", "4pt", "2pt", "AA", "3pt"):
        for bg, N in _configs(kind):
            lhs = sp.S(0)
            if include_bosons:
                lhs += bosonic_loop(bg, N)
            if include_ghosts:
                lhs += ghost_loop(bg, N)
            if include_fermions:
                lhs += fermions_loop(bg, N)
            ops = _ops(bg)
            rhs = sum(c * ops["O" + str(c)[1:]] for c in unknowns)
            eqs.append(sp.expand(lhs - rhs))
    # the equations are linear in c with polynomial-in-couplings right-hand sides
    sol = sp.solve(eqs, unknowns, dict=True)
    if len(sol) != 1:
        raise RuntimeError("matching system inconsistent or underdetermined")
    sol = sol[0]
    if any(c not in sol or sol[c].free_symbols & set(unknowns) for c in unknowns):
        raise RuntimeError(f"matching system underdetermined: {sol}")
    residual = [sp.simplify(e.subs(sol)) for e in eqs]
    if any(r != 0 for r in residual):
        raise RuntimeError(f"matching system inconsistent: {residual}")
    return {str(c): sp.factor(sol[c]) for c in unknowns}


def onshell(cs):
    """Reduce the Green's basis to {rho^3, rho |D Phi|^2} with the tree-level EOMs (4D units).

    Returns (c6 * pi^4/zeta3, c_kin * pi^4 T/zeta3) in the 3D normalisation of 2610.12284.
    """
    c = {k: sp.sympify(v) for k, v in cs.items()}
    cJ = GS**2 * c["c7"] - GS * c["c8"]          # coefficient of J^2 = 6 O2 + O4
    c2 = c["c2"] + 6 * cJ
    c4 = c["c4"] + cJ
    ckin = c2
    c6 = c["c1"] + 4 * LAMS * c4 + 4 * LAMS**2 * c["c6"]
    return sp.expand(c6), sp.expand(ckin)


PUBLISHED_C6 = (15 * GS**6 - 18 * GS**4 * LAMS + 75 * GS**2 * LAMS**2 + 100 * LAMS**3) / 480
PUBLISHED_CKIN = (51 * GS**4 + 200 * GS**2 * LAMS - 20 * LAMS**2) / 960


if __name__ == "__main__":
    import time
    t0 = time.time()
    cb = green_coefficients(include_fermions=False)
    print("bosonic Green's basis:", cb, f"({time.time() - t0:.0f} s)")
    c6b, ckb = onshell(cb)
    print("bosonic c6   :", sp.factor(c6b), " published:", sp.factor(PUBLISHED_C6))
    print("bosonic c_kin:", sp.factor(ckb), " published:", sp.factor(PUBLISHED_CKIN))
    t0 = time.time()
    cf = green_coefficients(include_bosons=False)
    print("fermionic Green's basis:", cf, f"({time.time() - t0:.0f} s)")
    c6f, ckf = onshell(cf)
    print("fermionic c6   :", sp.factor(c6f))
    print("fermionic c_kin:", sp.factor(ckf))
