"""The gravitational wave spectrum of one parameter point, for an external likelihood.

    from transitionlistener.spectrum import gw_spectrum
    h2Omega = gw_spectrum("models/TL_conformal_u1_fermions_3d.py", g=0.46, y=0.1, v_GeV=2.0)
    h2Omega(f_Hz)                    # h^2 Omega_GW today at the frequencies f_Hz
    h2Omega.observables              # the transition parameters it was computed from

The pipeline runs up to the spectrum and stops: no detector sensitivities and no PTA
likelihoods are computed. Every key a single-point run accepts can be passed, the
modelling choices of the spectrum (``gw_*``) and of the percolation (``percolation_*``) included.
The settings of the nucleation rate, the keys of ``nucleationConf``, are passed as the dict
``nucleation``. A point without a completed transition raises the error of the pipeline.

Documentation: https://tasillo.de/TransitionListener/

Authors:
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def _potential_class(modelfile: str, name: str):
    """The potential class of a model file, loaded once per process."""
    from transitionlistener.helper_functions import load_potential

    directory = os.path.dirname(modelfile)
    if directory not in sys.path:      # model files import their base model by name
        sys.path.insert(0, directory)
    return load_potential(modelfile, name)


class GWSpectrum:
    """h^2 Omega_GW(f) today of one point, callable on frequencies in Hz.

    ``observables`` holds the transition parameters the spectrum was computed from
    (percolation temperature, alpha, beta/H, reheating temperature, efficiencies).
    """

    def __init__(self, spectrum, observables: dict):
        self._spectrum = spectrum
        self.observables = observables

    def __call__(self, f_Hz, source: str = "all"):
        return self._spectrum.h2Omega_0_sum(np.asarray(f_Hz, dtype=float), source=source)


def gw_spectrum(modelfile: str, potential: str = "specific_potential", timeout: float = -1,
                nucleation: dict | None = None, **params) -> GWSpectrum:
    """The GW spectrum of the point ``params`` of the model in ``modelfile``.

    ``nucleation`` sets keys of ``nucleationConf`` on the model before the run (for example
    ``{"include_determinants": False}`` for the leading-order 3D rate). ``timeout`` in seconds,
    -1 for none; it uses ``signal.alarm`` and works only in the main thread. Raises the
    pipeline's error if the point has no completed transition.
    """
    from transitionlistener.gwfopt import FOPTspectrum
    from transitionlistener.interface.pipeline import _compute_single_point

    pot = _potential_class(os.path.abspath(modelfile), potential)
    if nucleation:
        from transitionlistener.config import NucleationConf
        unknown = set(nucleation) - {k for k in vars(NucleationConf) if not k.startswith("_")}
        if unknown:
            raise ValueError(f"unknown nucleationConf keys {sorted(unknown)}")
        model = pot

        def pot(*args, **kwargs):
            instance = model(*args, **kwargs)
            for key, value in nucleation.items():
                setattr(instance.config.nucleationConf, key, value)
            return instance
    context, err = _compute_single_point(params, pot, timeout, False, False, max_stage=3)
    if err is not None:
        raise err
    observables = context["strongest_transition_observables"]
    spectrum = FOPTspectrum(observables, {"A": 0.0, "gamma": 0.0}, False, config=context["pot"].config)
    return GWSpectrum(spectrum, observables)
