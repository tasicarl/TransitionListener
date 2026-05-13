"""
Particle and mass spectrum utilities with concrete particle species.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

ThermalPrefactor = float | Callable[[np.ndarray], np.ndarray]


def _ensure_callable(name: str, fn):
    """Return ``fn`` unchanged when callable and raise a targeted error otherwise."""
    if fn is None or callable(fn):
        return fn
    raise TypeError(f"{name} must be callable or None.")

@dataclass
class Scalar:
    """Structure to represent scalars"""
    name: str
    latex_name: str
    dof: float
    is_SM: bool
    gauge_coupling: float = 0.0
    is_physical: bool = True
    c: float = 1.5

@dataclass
class Goldstone:
    """Structure to represent Golstone bosons"""
    name: str
    latex_name: str
    dof: float
    is_SM: bool
    gauge_coupling: float = 0.0
    is_physical: bool = False
    c: float = 1.5


@dataclass
class GaugeBoson:
    """Structure to represent gauge bosons"""
    name: str
    latex_name: str
    dof: float
    is_SM: bool
    gauge_coupling: float = 0.0
    is_physical = True
    c: float = 5.0/6.0


@dataclass
class Fermion:
    """Structure to represent fermions."""
    name: str
    latex_name: str
    dof: float
    is_SM: bool
    c: float = 1.5


class MassSpectrum:
    """Container describing the particle content and mass functions of a model.

    The class keeps the legacy ``CosmoTransitions``-style tuple interface used
    throughout TransitionListener.  It stores particle metadata such as degrees
    of freedom and renormalisation constants alongside callables returning the
    field- and temperature-dependent mass eigenvalues.
    """

    def __init__(self, scalars, gaugeBosons=[], fermions=[],
                 boson_massSq_fn=None, fermion_massSq_fn=None):
        r"""Register the bosonic and fermionic sectors of the model.

        Parameters
        ----------
        scalars, gaugeBosons, fermions:
            Particle descriptors providing names, degrees of freedom, and the
            renormalization constant ``c`` used in the one-loop effective
            potential for both bosons and fermions.
        boson_massSq_fn:
            Callable returning the bosonic mass-squared spectrum
            :math:`m_i^2(\phi, T)`.
        fermion_massSq_fn:
            Callable returning the fermionic mass-squared spectrum
            :math:`m_f^2(\phi)`.
        """
        self.Nscalars = len(scalars)
        self.NgaugeBosons = len(gaugeBosons)
        self.Nbosons = self.Nscalars + self.NgaugeBosons
        self.Nfermions = len(fermions)
        self.bosons = scalars + gaugeBosons
        self.fermions = fermions

        # Necessary for correct DOFs
        self.number_gauge_bosons = self.calc_N_gauge_bosons(gaugeBosons)

        self.boson_massSq_fn = self.set_bmass_fn(boson_massSq_fn)
        self.fermion_massSq_fn = self.set_fmass_fn(fermion_massSq_fn)

        if fermions == []:
            self.dof_fermions = np.array([0.0])
            self.c_fermions = np.array([3./2.])
            self.is_SM_fermions = np.array([False])
        else:
            self.dof_fermions = np.array([particle.dof for particle in self.fermions])
            self.c_fermions = np.array([particle.c for particle in self.fermions])
            self.is_SM_fermions = np.array([particle.is_SM for particle in self.fermions])

        self.dof_bosons = np.array([particle.dof for particle in self.bosons])
        self.c_bosons = np.array([particle.c for particle in self.bosons])
        self.is_physical_bosons = np.array([particle.is_physical for particle in self.bosons])
        self.is_SM_bosons = np.array([particle.is_SM for particle in self.bosons])

        self.boson_gauge_couplings = np.zeros(self.Nbosons)
        for i in range(self.Nbosons):
            self.boson_gauge_couplings[i] = self.bosons[i].gauge_coupling

    def calc_N_gauge_bosons(self, gaugeBosons):
        """Infer the number of vector bosons from the total bosonic degrees of freedom."""
        dof = 0
        for gb in gaugeBosons:
            dof += gb.dof

        nGBosons = int(dof/3)
        resid = dof % 3
        if resid != 0:
            raise Exception("DOF of bosons does not add up!")
        return nGBosons

    def boson_labels(self, type: str):
        """Return boson names either in plain text or in LaTeX form."""
        if type == "latex":
            return [p.latex_name for p in self.bosons]
        elif type == "text":
            return [p.name for p in self.bosons]
        else:
            raise Exception("Wrong label type supplied.")

    def fermion_labels(self, type: str):
        """Return fermion names either in plain text or in LaTeX form."""
        if type == "latex":
            return [p.latex_name for p in self.fermions]
        elif type == "text":
            return [p.name for p in self.fermions]
        else:
            raise Exception("Wrong label type supplied.")

    def set_bmass_fn(self, bmass_fn):
        """Normalise the bosonic mass callback and supply a massless default when absent."""
        if bmass_fn is None:
            def fn(X, T):
                X = np.asarray(X)
                M2 = np.zeros(X[...,0].shape + (1, ))
                return M2
            return fn
        return bmass_fn

    def set_fmass_fn(self, fmass_fn):
        """Normalise the fermionic mass callback and supply a massless default when absent."""
        if fmass_fn is None:
            def fn(X):
                X = np.asarray(X)
                M2 = np.zeros(X[...,0].shape + (1, ))
                return M2
            return fn
        return fmass_fn

    def bosons_massSq(self, X: np.ndarray, T: float):
        r"""Return the bosonic masses entering the one-loop effective potential.

        The returned tuple is interpreted as ``(m_i^2, n_i, c_i, physical_i)``,
        where :math:`m_i^2(\phi, T)` are the bosonic mass eigenvalues,
        :math:`n_i` their degrees of freedom, and :math:`c_i` the
        renormalisation constants used in

        .. math::
           V_1^{B}(\phi, T=0)
           = \sum_i \frac{n_i\, m_i^4(\phi, 0)}{64\pi^2}
             \left[\log\!\left(\frac{m_i^2(\phi, 0)}{\mu^2}\right) - c_i\right].

        Parameters
        ----------
        X:
            Field configuration :math:`\phi`.
        T:
            Temperature in the internal TransitionListener units.

        Returns
        -------
        tuple
            ``(masses_sq, dof, c_constants, is_physical)``.
        """
        M2 = self.boson_massSq_fn(X, T)
        return M2, self.dof_bosons, self.c_bosons, self.is_physical_bosons

    def fermion_massSq(self, X: np.ndarray):
        r"""Return the fermionic masses entering the one-loop effective potential.

        The returned tuple is interpreted as ``(m_f^2, n_f)`` and is used in

        .. math::
           V_1^{F}(\phi)
           = -\sum_f \frac{n_f\, m_f^4(\phi)}{64\pi^2}
             \left[\log\!\left(\frac{m_f^2(\phi)}{\mu^2}\right) - \tfrac{3}{2}\right].

        Parameters
        ----------
        X:
            Field configuration :math:`\phi`.

        Returns
        -------
        tuple
            ``(masses_sq, dof)``.
        """
        M2 = self.fermion_massSq_fn(X)
        return M2, self.dof_fermions

    def get_gauge_coupling(self):
        """Return the per-boson gauge couplings associated with the stored spectrum."""
        return self.boson_gauge_couplings

        
@dataclass
class BaseParticle:
    """Shared particle descriptor used by the newer block-based spectrum API."""

    name: str
    latex_name: str
    dof: float
    is_SM: bool
    c: float
    gauge_coupling: float = 0.0
    thermal_prefactor: ThermalPrefactor = None
    hard_thermal_mass_fn: Optional[Callable[[np.ndarray, float], np.ndarray]] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the basic particle metadata and normalise optional callables."""
        if self.dof < 0:
            raise ValueError("Particle degrees of freedom must be non-negative.")
        self.hard_thermal_mass_fn = _ensure_callable(
            f"hard_thermal_mass_fn for {self.name}", self.hard_thermal_mass_fn
        )
        if self.thermal_prefactor is not None and not callable(self.thermal_prefactor):
            value = float(self.thermal_prefactor)
            self.thermal_prefactor = lambda X, value=value: np.array(value, dtype=float)

    def evaluate_prefactor(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Evaluate the optional thermal prefactor at field point ``X``."""
        if self.thermal_prefactor is None:
            return None
        return np.asarray(self.thermal_prefactor(np.asarray(X)), dtype=float)

    @property
    def is_BSM(self) -> bool:
        """Whether the particle belongs to the beyond-the-Standard-Model sector."""
        return not self.is_SM

    @property
    def statistic(self) -> str:
        """Bosonic/fermionic label implemented by concrete subclasses."""
        raise NotImplementedError

    @property
    def is_physical(self) -> bool:
        """Whether the particle should contribute as a physical excitation."""
        return True


@dataclass
class BosonSector:
    """Evaluated bosonic spectrum together with particle metadata."""

    masses_sq: np.ndarray
    dof: np.ndarray
    c: np.ndarray
    is_physical: np.ndarray
    gauge_coupling: np.ndarray
    is_bsm: np.ndarray
    particles: Sequence[BaseParticle]
    thermal_masses_sq: Optional[np.ndarray] = None

    def as_tuple(self) -> Tuple[np.ndarray, ...]:
        """Return the legacy tuple representation used by older callers."""
        return (
            self.masses_sq,
            self.dof,
            self.c,
            self.is_physical,
            self.gauge_coupling,
            self.is_bsm,
        )

    @property
    def latex_labels(self) -> List[str]:
        """LaTeX labels for each bosonic degree of freedom."""
        return [p.latex_name for p in self.particles]

    @property
    def text_labels(self) -> List[str]:
        """Plain-text labels for each bosonic degree of freedom."""
        return [p.name for p in self.particles]


@dataclass
class FermionSector:
    """Evaluated fermionic spectrum together with particle metadata."""

    masses_sq: np.ndarray
    dof: np.ndarray
    is_bsm: np.ndarray
    particles: Sequence[BaseParticle]

    def as_tuple(self) -> Tuple[np.ndarray, ...]:
        """Return the legacy tuple representation used by older callers."""
        return (self.masses_sq, self.dof, self.is_bsm)

    @property
    def latex_labels(self) -> List[str]:
        """LaTeX labels for each fermionic degree of freedom."""
        return [p.latex_name for p in self.particles]

    @property
    def text_labels(self) -> List[str]:
        """Plain-text labels for each fermionic degree of freedom."""
        return [p.name for p in self.particles]


@dataclass
class SpectrumSnapshot:
    """Simultaneous bosonic and fermionic spectrum evaluation at a given state."""

    bosons: BosonSector
    fermions: FermionSector


@dataclass
class MassBlock:
    """Group of particles sharing one mass function and, optionally, one thermal correction."""

    particles: Sequence[BaseParticle]
    mass_function: Callable[[np.ndarray, float], np.ndarray]
    hard_thermal_mass_function: Optional[Callable[[np.ndarray, float], np.ndarray]] = None
    label: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate particle statistics and normalise the supplied callables."""
        if not self.particles:
            raise ValueError("MassBlock requires at least one particle.")
        stats = {p.statistic for p in self.particles}
        if len(stats) != 1:
            raise ValueError("All particles in a MassBlock must share the same statistic.")
        self.particles = list(self.particles)
        self.mass_function = _ensure_callable("mass_function", self.mass_function)
        self.hard_thermal_mass_function = _ensure_callable(
            "hard_thermal_mass_function", self.hard_thermal_mass_function
        )

    @property
    def statistic(self) -> str:
        """Bosonic/fermionic statistic shared by every particle in the block."""
        return self.particles[0].statistic

    def evaluate(self, X: np.ndarray, T: float) -> np.ndarray:
        """Evaluate the block mass function and align its trailing axis with ``particles``."""
        masses = np.asarray(self.mass_function(X, T))
        if masses.shape[-1] != len(self.particles):
            if masses.shape[0] == len(self.particles):
                masses = np.moveaxis(masses, 0, -1)
        if masses.shape[-1] != len(self.particles):
            raise ValueError(
                f"Mass function for block {self.label!r} returned shape {masses.shape}, "
                f"expected trailing dimension {len(self.particles)}."
            )
        return masses

    def evaluate_thermal(self, X: np.ndarray, T: float) -> Optional[np.ndarray]:
        r"""Evaluate hard thermal masses or ``c_i T^2``-style prefactors for the block.

        When no explicit thermal-mass callback is provided, the function falls
        back to per-particle prefactors and returns

        .. math::
           \Pi_i(\phi, T) = c_i(\phi)\, T^2.
        """
        if self.hard_thermal_mass_function is not None:
            thermal = np.asarray(self.hard_thermal_mass_function(X, T))
            if thermal.shape[-1] != len(self.particles):
                if thermal.shape[0] == len(self.particles):
                    thermal = np.moveaxis(thermal, 0, -1)
            if thermal.shape[-1] != len(self.particles):
                raise ValueError(
                    f"Thermal mass function for block {self.label!r} returned shape {thermal.shape}, "
                    f"expected trailing dimension {len(self.particles)}."
                )
            return thermal

        prefactors = []
        has_prefactor = False
        for particle in self.particles:
            pref = particle.evaluate_prefactor(X)
            if pref is None:
                pref = 0.0
            else:
                has_prefactor = True
            prefactors.append(np.asarray(pref, dtype=float))
        if not has_prefactor:
            return None
        pref_array = np.stack(prefactors, axis=-1)
        T_arr = np.asarray(T, dtype=float)
        return pref_array * (np.square(T_arr)[..., np.newaxis])

__all__ = [
    "Scalar",
    "Goldstone",
    "GaugeBoson",
    "Fermion",
    "MassSpectrum",
    "MassBlock",
    "BosonSector",
    "FermionSector",
    "SpectrumSnapshot",
]
