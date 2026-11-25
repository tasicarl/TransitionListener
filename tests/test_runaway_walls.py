"""The efficiencies of walls that collide before the friction stops them, and the spectrum of the two kinds of shell."""
import numpy as np
import pytest

from transitionlistener import hydrodynamics
from transitionlistener.config import Configuration


def kappas(alpha, alpha_inf, alpha_eq, Rsep, R0, mode="NLO"):
    config = Configuration().gwConf
    config.bw_collisions = mode
    return hydrodynamics.calc_kappas(alpha, alpha_inf, alpha_eq, 1.0, 0.57, Rsep, R0, config)


def test_the_efficiency_is_the_broken_power_law_at_the_effective_radius():
    """kappa = K / (1 + R_eff / R_eq), with R_eq = (3/2) R0 gamma_eq and R_eff = 5 / beta."""
    Rsep, R0, alpha, alpha_inf, alpha_eq = 1e14, 1.0, 1e16, 1e9, 1e5
    gamma_eq = (alpha - alpha_inf) / alpha_eq
    R_eff = 5.0 / (8 * np.pi) ** (1 / 3) * Rsep
    expected = (1 - alpha_inf / alpha) * (1 - 1 / gamma_eq) / (1 + R_eff / (1.5 * R0 * gamma_eq))
    phi, sw, _ = kappas(alpha, alpha_inf, alpha_eq, Rsep, R0)
    assert phi == pytest.approx(expected, rel=1e-12)
    assert phi + sw == pytest.approx(1.0, rel=1e-5)


def test_half_of_the_energy_is_in_the_wall_where_the_effective_radius_is_r_eq():
    gamma_eq = (1e20 - 1e9) / 1e5
    Rsep = 1.5 * gamma_eq * (8 * np.pi) ** (1 / 3) / 5.0
    assert kappas(1e20, 1e9, 1e5, Rsep, 1.0)[0] == pytest.approx(0.5, rel=1e-6)


def test_walls_that_collide_long_before_r_eq_carry_nearly_all_of_the_energy():
    phi, sw, _ = kappas(1e20, 1e9, 1e5, 1e6, 1.0)
    assert phi == pytest.approx(1.0, abs=1e-6) and 0.0 <= sw < 1e-6


def test_walls_that_reach_r_eq_long_before_they_collide_fall_as_one_over_r():
    assert kappas(1e16, 1e9, 1e5, 2e14, 1.0)[0] == pytest.approx(0.5 * kappas(1e16, 1e9, 1e5, 1e14, 1.0)[0], rel=1e-3)


def test_friction_that_stops_the_walls_leaves_them_no_energy():
    assert kappas(1e8, 1e9, 1e5, 1e6, 1.0)[0] == 0


def test_without_a_critical_bubble_there_is_no_answer():
    assert all(np.isnan(k) for k in kappas(1e16, 1e9, 1e5, 1e14, np.nan))
