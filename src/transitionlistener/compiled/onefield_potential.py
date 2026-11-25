"""The compiled potential of a one-field model, read from the model itself.

generic_potential calls :func:`build` at the end of the construction of a model with one field. The form the kernels of
compiled.onefield can evaluate, and how each piece follows from the functions of the model::

    V0 + Vct             a polynomial in phi up to phi^6: the coefficients from its values at seven fields;
    boson masses         m^2 = a0 + a1 phi + a2 phi^2 + cT T^2 for each mode: the coefficients from three fields and
                         one temperature;
    fermion masses       m^2 = a0 + a1 phi + a2 phi^2;
    V1                   sum_i (+-) n_i M_i^4 (ln|M_i^2| - k_i) / (64 pi^2), with M_i^2 = m_i^2 + goldstone_shift for a
                         Goldstone boson: the constants k_i are read from the spectrum
                         and the renormalisation scale, or, for a model with its own V1, solved for from its values.

Masses from a mass matrix with mixing, a tree level with other powers or logarithms, a thermal mass that depends on the
field, or Standard Model fields in the energy density do not have this form. The result is compared
with the Python potential on a grid of fields and temperatures from 1e-8 of the field scale upwards, so a model outside
the form runs on the Python potential, with a warning (generic_potential._update_kernels).

Documentation: https://tasillo.de/TransitionListener/

Authors:
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""
import numpy as np
from scipy import interpolate

from transitionlistener import errors, finiteT, helper_functions
from transitionlistener import thermodynamics as td
from transitionlistener.compiled import onefield as kernels
from transitionlistener.compiled.roots import CONVERGED
from transitionlistener.compiled.splines import _der_coef, _knot_table, _piece_table, _ppeval

_DAISY = {"off": kernels.DAISY_OFF, "Parwani": kernels.DAISY_ALL_MODES, "ArnoldEspinosa": kernels.DAISY_ZERO_MODES}
#: The criteria of bubbledynamics for the temperature of the broken phase, in the numbering of kernels._criterion.
TB_CRITERION, ENERGY_CRITERION, ENTROPY_CRITERION = 0, 1, 2

_tables = {}


def _shared_tables():
    """The arrays of the thermal functions and of the tabulated degrees of freedom, once per process."""
    if not _tables:
        arrays = lambda *a: tuple(np.ascontiguousarray(x, dtype=float) for x in a)     # noqa: E731
        tb, cb, tf, cf = arrays(finiteT._tckb[0], finiteT._tckb[1], finiteT._tckf[0], finiteT._tckf[1])
        tabb, tabf = _knot_table(tb), _knot_table(tf)
        _tables["P"] = (tb, cb, tabb, tf, cf, tabf)
        _tables["pieces"] = (tb, _piece_table(tb, cb), tabb, tf, _piece_table(tf, cf), tabf)
        _tables["limits"] = (finiteT._xbmin, finiteT._xbmax, finiteT._xfmin, finiteT._xfmax,
                             finiteT.Jb_spline(np.array([finiteT._xbmin]))[0],
                             finiteT.Jf_spline(np.array([finiteT._xfmin]))[0])
        _tables["J"] = (tb, cb, tabb, _der_coef(tb, cb), tf, cf, tabf, _der_coef(tf, cf),
                        np.array([float(interpolate.splev(finiteT._xbmin, finiteT._tckb, der=1)),
                                  float(interpolate.splev(finiteT._xfmin, finiteT._tckf, der=1))]))
        counts = []
        for spline in (td.Ieb, td.Ipb, td.Ief, td.Ipf):
            counts += [*arrays(spline._spline.t, np.ravel(spline._spline.c)), float(spline.x[0]), float(spline.x[-1])]
        _tables["G"] = tuple(counts)
        _tables["e"] = arrays(td.Ie_geffSM_smooth.c, td.Ie_geffSM_smooth.x)
        _tables["s"] = arrays(td.Is_geffSM_smooth.c, td.Is_geffSM_smooth.x)
    return _tables


class OneFieldPotential:
    """The kernels of one model, with the calling conventions of generic_potential.

    Each method answers for the inputs it covers and returns None for the others, which then take the generic code.
    It is built anew when the model changes (generic_potential._update_kernels), so everything that is cheap to read
    once is read once, in :func:`build`."""

    def __init__(self, table, daisy, order, T_eps, gradient, hessian):
        shared = _shared_tables()
        self.C = table
        self.P = (table, *shared["P"])
        self.daisy = daisy
        self.order = order
        self.T_eps = float(T_eps)
        self.gdx, self.gcoef = gradient
        self.hdx, self.hcoef = hessian
        self.radiation = {}
        self.J, self.G, self.ce, self.cs = shared["J"], shared["G"], shared["e"], shared["s"]
        self.thermal = None     # see _prepare_thermodynamics

    def _radiation(self, pot, T, include_decoupled):
        key = (T, include_decoupled)
        value = self.radiation.get(key)
        if value is None:
            value = float(np.asarray(pot.constantTerms(np.asanyarray(T), include_decoupled=include_decoupled)))
            if len(self.radiation) > 4096:
                self.radiation.clear()
            self.radiation[key] = value
        return value

    def vtot(self, pot, X, T, include_radiation, include_decoupled):
        Ta = np.asanyarray(T)
        if type(Ta) is not np.ndarray or Ta.dtype != np.float64:
            return None
        Xa = X if (type(X) is np.ndarray and X.dtype == np.float64) else np.asarray(X, dtype=float)
        if Xa.ndim < 1 or Xa.shape[-1] != 1:
            return None
        if Ta.ndim == 0:
            Tf = float(Ta)
            if Xa.ndim == 1:
                y = kernels.veff(float(Xa[0]), Tf, self.P, self.daisy)
                if include_radiation:
                    y += self._radiation(pot, Tf, bool(include_decoupled))
                return np.float64(y)
            shape = Xa.shape[:-1]
            field = np.ascontiguousarray(Xa.reshape(-1))
            y = kernels.vtot_many(field, np.full(field.size, Tf), self.P, self.daisy).reshape(shape)
            if include_radiation:
                y += self._radiation(pot, Tf, bool(include_decoupled))
            return y
        shape = np.broadcast_shapes(Xa.shape[:-1], Ta.shape)
        field = np.ascontiguousarray(np.broadcast_to(Xa[..., 0], shape)).reshape(-1)
        temperature = np.ascontiguousarray(np.broadcast_to(Ta, shape)).reshape(-1)
        y = kernels.vtot_many(field, temperature, self.P, self.daisy).reshape(shape)
        if include_radiation:
            y += pot.constantTerms(Ta, include_decoupled=include_decoupled)
        return y

    def gradv(self, X, T):
        if not (type(T) is float or np.ndim(T) == 0):
            return None
        Xa = X if (type(X) is np.ndarray and X.dtype == np.float64) else np.asarray(X, dtype=float)
        if Xa.ndim < 1 or Xa.shape[-1] != 1:
            return None
        field = np.ascontiguousarray(Xa.reshape(-1))
        return kernels.gradv_many(field, float(T), self.P, self.daisy, self.gdx, self.gcoef).reshape(Xa.shape)

    def d2v(self, X, T):
        Xa = X if (type(X) is np.ndarray and X.dtype == np.float64) else np.asarray(X, dtype=float)
        if Xa.shape != (1,) or not (type(T) is float or np.ndim(T) == 0):
            return None
        return np.array([[kernels.d2v(float(Xa[0]), float(T), self.P, self.daisy, self.hdx, self.hcoef)]])

    def dgradv_dt(self, X, T):
        Xa = X if (type(X) is np.ndarray and X.dtype == np.float64) else np.asarray(X, dtype=float)
        if Xa.ndim < 1 or Xa.shape[-1] != 1:
            return None
        shape = np.broadcast_shapes(Xa.shape[:-1], np.shape(T))
        field = np.ascontiguousarray(np.broadcast_to(Xa[..., 0], shape)).reshape(-1)
        temperature = np.ascontiguousarray(np.broadcast_to(np.asarray(T, dtype=float), shape)).reshape(-1)
        out = kernels.dgradv_dt_many(field, temperature, self.T_eps, self.order, self.P, self.daisy, self.gdx, self.gcoef)
        return out.reshape(shape + (1,))

    def minimum(self, x, T, xtol):
        """optimize.fmin of Vtot(X, T, False, False) from x, or None where the search ends without an answer."""
        ok, found = kernels.nelder_mead(float(x[0]), float(T), xtol, self.P, self.daisy)
        return np.array([found]) if ok else None

    def newton_minimum(self, X, T, phitol, step_size, origin):
        """phases.findLocalMinimum. ``origin`` is what generic_potential._origin_stencil gives."""
        Xa = np.asarray(X)
        if Xa.shape != (1,) or Xa.dtype != np.float64 or not (type(T) is float or np.ndim(T) == 0):
            return None
        mass, om, ostep = origin if origin is not None else (0.0, -1.0, -1.0)
        found = kernels.newton_minimum(float(Xa[0]), float(T), self.P, self.daisy, self.hdx, self.hcoef, self.gdx,
                                       self.gcoef, float(phitol), float(step_size), mass, om, ostep)
        return np.array([found])

    # ------------------------------------------------------------------------- thermodynamics along a traced phase
    def _prepare_thermodynamics(self, pot):
        """The numbers of the kernels for the energy and entropy density, unless a model replaces the generic
        expressions (see overrides in :func:`build`) or holds Standard Model fields."""
        spectrum = pot.mass_spectrum
        if np.any(spectrum.is_SM_bosons) or np.any(spectrum.is_SM_fermions):
            return
        Ng = spectrum.number_gauge_bosons
        self.thermal = dict(Ng=float(Ng), ghosts=float(td.s_geff(0.0, 1.0, Ng, "b")),
                            V0ref=float(pot.V0(pot.X0) + pot.Vct(pot.X0) + pot.V1_from_X(pot.X0)),
                            CF=float(pot.conversionFactor), coupled=pot.kin_coupled_e_geff,
                            decoupled=pot.kin_decoupled_e_geff, table=pot.kin_coupled_e_geff is td.e_geffSM,
                            sm_bath=(pot.kin_coupled_e_geff is td.e_geffSM and pot.kin_coupled_p_geff is td.p_geffSM))

    def _baths(self, t, include_decoupled):
        th = self.thermal
        CF = th["CF"]
        coupled = 0.0 if th["table"] else float(th["coupled"](t, CF))
        lg = float(np.log10(t * CF)) if th["table"] else 0.0
        return coupled, th["table"], lg, float(th["decoupled"](t, CF)) if include_decoupled else 0.0

    def energy_density(self, X, T, include_decoupled):
        """generic_potential.energyDensity at one field point and a positive temperature."""
        th = self.thermal
        if th is None or not (type(T) is float or type(T) is np.float64) or not T > 0.0:
            return None
        Xa = np.asarray(X)
        if Xa.shape != (1,) or Xa.dtype != np.float64:
            return None
        t = float(T)
        return kernels.energy_density(float(Xa[0]), t, *self._baths(t, include_decoupled), th["Ng"], th["V0ref"],
                                      *self.ce, self.C, self.J)

    def energy_density_on_phase(self, spline, t, include_decoupled):
        th = self.thermal
        return kernels.energy_density_on_phase(t, *self._baths(t, include_decoupled), th["Ng"], th["V0ref"], *spline,
                                               *self.ce, self.C, self.J)

    def h_eff(self, spline, t):
        return kernels.h_eff(t, self.thermal["ghosts"], *spline, self.C, self.G)

    def h_eff_coupled(self, t):
        return _ppeval(*self.cs, float(np.log10(t * self.thermal["CF"])), True)

    def entropy_history(self, spline, T, Tmin, Tmax, rel, sound_speed_sq, entropy):
        th = self.thermal
        return kernels.entropy_and_sound_speed(T, Tmin, Tmax, rel, th["ghosts"], *spline, self.C, self.G, *self.cs,
                                               th["CF"], sound_speed_sq, entropy)

    def criterion_root(self, kind, spline, a, b, constants):
        """(root, converged) of a criterion on the tabulated bath, with the numbers that do not depend on the root."""
        th = self.thermal
        root, status = kernels.criterion_root(a, b, kind, *constants, th["Ng"], th["V0ref"], th["ghosts"], *spline,
                                              *self.ce, *self.cs, self.C, self.J, self.G, th["CF"])
        return root, status == CONVERGED

    def entropy_on_phase(self, spline, t):
        th = self.thermal
        return kernels.entropy(t, th["ghosts"], *spline, self.C, self.G, *self.cs, th["CF"])


def _snap(values, scale):
    """Zero what is rounding: a coefficient 1e-13 below the scale would dominate at small field."""
    values = np.asarray(values, dtype=float)
    return np.where(np.abs(values) < 1e-13 * scale, 0.0, values)


def _mode_table(pot, s, generic_one_loop):
    """(a0, a1, a2, cT, n, k) of the bosons, (a0, a1, a2, n, k) of the fermions, and the polynomial of V0 + Vct."""
    def bosons(phi, T):
        return np.asarray(pot.boson_massSq(np.array([phi]), np.asarray(T))[0], dtype=float).reshape(-1)

    def fermions(phi):
        return np.asarray(pot.fermion_massSq(np.array([phi]))[0], dtype=float).reshape(-1)

    def coefficients(f):
        m0, up, down = f(0.0), f(s), f(-s)
        size = float(np.max(np.abs([m0, up, down]))) if m0.size else 0.0
        return _snap(m0, size), _snap((up - down) / (2 * s), size / s), _snap((up + down - 2 * m0) / (2 * s * s), size / s**2)

    a0, a1, a2 = coefficients(lambda phi: bosons(phi, 0.0))
    cT = _snap((bosons(0.0, s) - bosons(0.0, 0.0)) / (s * s), np.max(np.abs(a2)) + 1e-300)
    f0, f1, f2 = coefficients(fermions)
    spectrum = pot.boson_massSq(np.array([s]), np.asarray(0.0))
    nb = np.asarray(spectrum[1], dtype=float).reshape(-1)
    nf = np.asarray(pot.fermion_massSq(np.array([s]))[1], dtype=float).reshape(-1)
    shift = pot.goldstone_shift * (1.0 - np.asarray(spectrum[3], dtype=float).reshape(-1))

    x = np.cos(np.pi * np.arange(7) / 6.0)
    field = (x * s)[:, np.newaxis]
    tree = np.asarray(pot.V0(field) + pot.Vct(field) * np.ones(7), dtype=float)
    size = float(np.max(np.abs(tree)))
    poly = _snap(np.polyfit(x, tree, 6)[::-1], size) / s ** np.arange(7)

    if generic_one_loop:
        logscale = np.log(pot.renormScaleSq)
        k = np.concatenate((np.asarray(spectrum[2], dtype=float).reshape(-1) + logscale, np.full(nf.size, 1.5 + logscale)))
        return np.column_stack((a0, a1, a2, cT, nb, k[:nb.size], shift)), np.column_stack((f0, f1, f2, nf, k[nb.size:])), poly

    phi = s * np.geomspace(1e-2, 2.0, 24)
    m2b = a0[:, None] + phi * (a1[:, None] + phi * a2[:, None])
    m2f = f0[:, None] + phi * (f1[:, None] + phi * f2[:, None])
    fw = nf[:, None] * m2f * m2f
    bw = nb[:, None] * m2b * m2b

    def logs(m2):
        return np.log(np.where(m2 != 0.0, np.abs(m2), 1.0))

    known = np.sum(bw * logs(m2b), axis=0) - np.sum(fw * logs(m2f), axis=0)
    one_loop = np.asarray(pot.V1_from_X(phi[:, None]), dtype=float)
    columns = np.vstack((-bw, fw)).T
    norm = np.max(np.abs(nb)) * max(float(np.max(np.abs(a2))) * s * s + float(np.max(np.abs(a0))), 1e-300) ** 2
    k = np.linalg.lstsq(columns / norm, (64 * np.pi * np.pi * one_loop - known) / norm, rcond=1e-13)[0]
    return (np.column_stack((a0, a1, a2, cT, nb, k[:nb.size], np.zeros(nb.size))),
            np.column_stack((f0, f1, f2, nf, k[nb.size:])), poly)


def _agrees(compiled, pot, s):
    """The kernels against the Python potential, over the range of fields and temperatures of a run."""
    python = pot._Vtot_generic
    reference = 1e-3 * s
    for phi in s * np.array([-0.7, 0.0, 1e-5, 1e-2, 0.3, 1.0, 1.7]):
        for T in s * np.array([1e-8, 1e-3, 0.1, 2.0]):
            field = np.array([phi])
            a = float(python(field, float(T), False, True))
            b = float(compiled.vtot(pot, field, float(T), False, True))
            size = abs(a) + T**4 + abs(float(python(field, reference, False, True)))
            if not abs(a - b) <= 1e-7 * size:
                return False
    return True


def build(pot, thermal_overrides, generic_one_loop):
    """The :class:`OneFieldPotential` of ``pot``. Raises NotImplementedError, saying why, for a model outside the
    compiled form. ``thermal_overrides`` is True if the model replaces the generic energy density, ``generic_one_loop``
    if ``V1`` is the generic one."""
    if pot.daisy not in _DAISY:
        raise NotImplementedError(f"no kernel for the daisy scheme {pot.daisy!r}")
    if pot.deriv_order not in (2, 4):
        raise NotImplementedError(f"no kernel for derivatives of order {pot.deriv_order}")
    s = abs(float(np.ravel(getattr(pot, "X0", 0.0))[0])) or float(getattr(pot, "v_stable", 0.0) or 0.0)
    if not np.isfinite(s) or s <= 0.0:
        raise NotImplementedError("no field scale")
    bosons, fermions, poly = _mode_table(pot, s, generic_one_loop)
    if not (np.all(np.isfinite(bosons)) and np.all(np.isfinite(fermions)) and np.all(np.isfinite(poly))):
        raise NotImplementedError("the spectrum is not finite")
    limits = _shared_tables()["limits"]
    table = np.zeros(kernels.MODES + bosons.size + fermions.size)
    table[kernels.NBOS], table[kernels.NFER] = bosons.shape[0], fermions.shape[0]
    table[kernels.POLY:kernels.POLY + 7] = poly
    table[kernels.XBMIN:kernels.MODES] = limits
    table[kernels.MODES:kernels.MODES + bosons.size] = bosons.reshape(-1)
    table[kernels.MODES + bosons.size:] = fermions.reshape(-1)
    stencil = lambda a: np.ascontiguousarray(a, dtype=float)      # noqa: E731
    gradient = helper_functions.gradientFunction(None, pot.x_eps, 1, pot.deriv_order)
    hessian = helper_functions.hessianFunction(None, pot.x_eps, 1, pot.deriv_order)
    compiled = OneFieldPotential(table, _DAISY[pot.daisy], pot.deriv_order, pot.T_eps,
                                 (stencil(gradient.dx[:, 0, 0]), stencil(gradient.coef[:, 0])),
                                 (stencil(hessian.dx[0][0][:, 0]), stencil(hessian.coef[0][0])))
    if not _agrees(compiled, pot, s):
        raise NotImplementedError("the potential does not have the compiled form: masses squared quadratic in the "
                                  "field, tree level a polynomial up to phi^6")
    if not thermal_overrides:
        compiled._prepare_thermodynamics(pot)
    return compiled
