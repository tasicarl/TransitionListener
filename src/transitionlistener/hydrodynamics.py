"""This module contains routines to compute the hydrodynamics
of the expanding bubbles.

Based on the code snippet provided in `arxiv:2303.10171`.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

import numpy as np
from scipy.integrate import solve_ivp, simpson
from scipy.optimize import root_scalar

from transitionlistener import generic_potential
from transitionlistener.phases import PhaseInfo

from . import console
from rich import print
from rich.panel import Panel


def _local_temperature_step(pot, T: float) -> float:
    """Finite-difference temperature step that keeps the stencil at T > 0."""
    T_abs = abs(float(T))
    dT = max(float(getattr(pot, "T_eps", 1e-3)), T_abs * 1.0e-4)
    if T_abs > 0.0:
        dT = min(dT, 0.25 * T_abs)
    return dT


def kappa_sw(alpha : float, vw: float, cs: float) -> float:
    """Approximate the sound-wave efficiency factor using fit formulas from
    1004.4187 by Jose R. Espinosa, Thomas Konstandin, Jose M. No & 
    Geraldine Servant
    
    Parameters
    ----------
    alpha : float
        The strength of the phase transition at nucleation.
    vw : float
        The bubble wall velocity.
    cs : float
        The speed of sound in the broken phase.
    
    Returns
    -------
    float
        The sound-wave efficiency factor.
    """
    # Intermediate kappas, Jouguet velocity
    kA = vw**(6/5) * 6.9 * alpha / (1.36 - 0.037 * np.sqrt(alpha) + alpha)
    kB = alpha**(2/5) / (0.017 + (0.997 + alpha)**(2/5))
    kC = np.sqrt(alpha) / (0.135 + np.sqrt(0.98 + alpha))
    kD = alpha / (0.73 + 0.083 * np.sqrt(alpha) + alpha)
    dK = -0.9 * np.log(np.sqrt(alpha) / (1 + np.sqrt(alpha)))
    vJ = (cs + np.sqrt(alpha**2 + 2 * alpha / 3)) / (1 + alpha)
    lK = kC - kB - (vJ - cs) * dK

    if vw <= cs:  # Choose regime
        num = (cs**(11/5) * kA * kB)
        denom = ((cs**(11/5) - vw**(11/5)) * kB + vw * cs**(6/5) * kA)
        return num / denom
    elif vw < vJ:
        return kB + (vw - cs) * dK + ((vw - cs)**3 / (vJ - cs)**3) * lK
    else:
        num = (vJ - 1)**3 * vJ**(5/2) * vw**(-5/2) * kC * kD
        denom = ((vJ - 1)**3 - (vw - 1)**3) * vJ**(5/2) * kC + (vw - 1)**3 * kD
        return num / denom


class Hydrodynamics():
    """Provide hydrodynamic solutions for expanding first-order phase transition bubbles."""

    def __init__(self, pot : generic_potential, high_phase : PhaseInfo,
                 low_phase : PhaseInfo, verbose=False):
        """Store the potential and phase data used to solve the hydrodynamics problem.

        Parameters
        ----------
        pot : generic_potential
            The effective potential describing the plasma.
        high_phase : PhaseInfo
            Phase outside the bubble wall.
        low_phase : PhaseInfo
            Phase inside the bubble wall.
        verbose : bool, optional
            Toggle diagnostic logging for root finding and integration steps.
        """
        self.pot = pot
        self.high_phase = high_phase
        self.low_phase = low_phase
        self.verbose = verbose

    def calc_cs(self, T: float, sym: bool) -> float:
        """Calculate the speed of sound in the broken phase.

        Parameters
        ----------
        T : float
            Temperature, typically at nucleation
        sym : bool
            Whether to use symmetric or broken phase

        Returns
        ----------
        float
            speed of sound in the broken phase.
        """
        phase = self.high_phase if sym else self.low_phase
        dT = _local_temperature_step(self.pot, T)
        phi = phase.valAt(T)
        dVdT = self.pot.dVdT(phi, T, dT=dT, include_decoupled=False)
        ddVdT = self.pot.d2VdT2(phi, T, dT=dT, include_decoupled=False)
        cs = np.sqrt(dVdT/(T*ddVdT))
        return cs

    def calcWallVelocityLTE(self, Tn: float) -> float:
        """Calculate the wall velocity in local thermal equilibrium (LTE).
        This is done by solving the matching conditions across the bubble wall.
        It is based on the method described in `arxiv:2303.10171`.

        Whether the species directly coupled to the scalar undergoing the
        phase transition and the rest of the heat bath behave as a single
        fluid or as two fluids is read from
        ``pot.config.gwConf.coupled_hydrodynamics``. The bubble wall always
        does work on the species coupled to the transitioning scalar; the
        rest of the heat bath is only directly coupled to the wall when this
        flag is ``True``. (When it is ``False``, the two sectors can still be
        in thermal contact through indirect, slower processes; this switch
        only controls whether they are treated as a single hydrodynamic
        fluid in the matching conditions.)

        Parameters
        ----------
        Tn : float
            Temperature, typically at nucleation

        Returns
        ----------
        float
            wall velocity in local thermal equilibrium.
        """

        coupled = bool(self.pot.config.gwConf.coupled_hydrodynamics)
        dT = _local_temperature_step(self.pot, Tn)
        phi_s = self.high_phase.valAt(Tn)
        phi_b = self.low_phase.valAt(Tn)
        dVdT_s = self.pot.dVdT(phi_s, Tn, dT=dT, include_decoupled=coupled)
        dVdT_b = self.pot.dVdT(phi_b, Tn, dT=dT, include_decoupled=coupled)
        ddVdT_s = self.pot.d2VdT2(phi_s, Tn, dT=dT, include_decoupled=coupled)
        ddVdT_b = self.pot.d2VdT2(phi_b, Tn, dT=dT, include_decoupled=coupled)

        w_s = - Tn * dVdT_s
        w_b = - Tn * dVdT_b
        psi_n = w_b/w_s
        cs2 = dVdT_s/(Tn*ddVdT_s)
        cb2 = dVdT_b/(Tn*ddVdT_b)

        # Calculate the alpha definition from the paper
        # eq. (19) in `arxiv:2303.10171`
        p_s = - self.pot.Vtot(phi_s, Tn, include_decoupled=coupled)
        p_b = - self.pot.Vtot(phi_b, Tn, include_decoupled=coupled)

        e_s = w_s - p_s
        e_b = w_b - p_b

        DTheta = (e_s - p_s/cb2 - (e_b - p_b/cb2))
        alpha = DTheta / (3*w_s)

        vw = self.find_vw(alpha, cb2, cs2, psi_n)
        return vw

    def find_vJ(self, alN: float, cb2: float) -> float:
        """Calculate the Jouguet velocity, i.e. the smallest wall velocity a
        physically consistent detonation solution is allowed to have from the
        PT strength α and sound speed.

        Parameters
        ----------
        alN : float
            Strength of the phase transition at nucleation.
        cb2 : float
            Sound speed squared in the broken phase.

        Returns
        -------
        float
            The Jouguet velocity.
        """
        return np.sqrt(cb2) * (1 + np.sqrt(3 * alN * (1 - cb2 + 3 * cb2 * alN))) \
            / (1 + 3 * cb2 * alN)

    def get_vp(self, vm: float, al: float, cb2: float, branch=-1) -> float:
        """Find the veloity outside the bubble using the
        matching conditions.

        Parameters
        ----------
        vm : float
            The plasma velocity in the wall frame inside the bubble.
        al : float
            Strength of the phase transition at nucleation.
        cb2 : float
            The speed of sound squared in the broken phase.
        branch : int
            Choose the branch of the solution, either -1 or +1.
        Returns
        -------
        float
            The plasma velocity in the wall frame outside the bubble.
        """
        disc = vm**4 - 2*cb2 * vm**2*(1 - 6*al) + cb2**2*(1 - 12*vm**2 * al * (1 - 3*al))
        return 0.5 * (cb2 + vm**2 + branch * np.sqrt(disc))/(vm + 3*cb2 * vm * al)

    def w_from_alpha(self, al: float, alN: float, nu: float, mu: float) -> float:
        """Compute the enthalpy from alpha

        Parameters
        ----------
        al : float
            Strength of the phase transition at the wall.
        alN : float
            Strength of the phase transition at nucleation.
        nu : float
            Parameter depending on the speed of sound in the broken phase.
        mu : float
            Parameter depending on the speed of sound in the symmetric phase.

        Returns
        -------
        float
            The enthalpy in the wall frame.
        """
        return (abs((1 - 3 * alN) * mu - nu) +1e-100) / (abs ((1 -3* al) * mu - nu) +1e-100)

    def eqWall(self, al : float, alN: float, vm: float, nu: float,
               mu: float, psiN: float, solution: int = -1):
        """

        Parameters
        ----------
        al : float
            Strength of the phase transition at the wall.
        alN : float
            Strength of the phase transition at nucleation.
        vm : float
            Velocity of the fluid at the wall.
        nu : float
            Parameter depending on the speed of sound in the broken phase.
        mu : float
            Parameter depending on the speed of sound in the symmetric phase.
        psiN : float
            Ratio of enthalpies at nucleation.
        solution : int
            Choose the branch of the solution, either -1 or +1.

        Returns
        -------
        float
            The value of the matching condition equation.

        """
        vp = self.get_vp(vm, al ,1/(nu -1), solution)
        ga2m, ga2p = 1/(1 - vm**2) ,1/(1 - vp**2)
        psi = psiN * self.w_from_alpha(al, alN, nu, mu)**(nu / mu -1)
        return vp * vm * al /(1 -(nu -1) * vp * vm) -(1 -3* al -(ga2p / ga2m)**(nu /2) *
                                                    psi) /(3* nu)

    def solve_alpha(self, vw, alN, cb2, cs2, psiN):
        """Find the strength of the phase transition at the wall.

        Parameters
        ----------
        vw : float
            Velocity of the fluid at the wall.
        alN : float
            Strength of the phase transition at nucleation.
        cb2 : float
            Speed of sound squared in the broken phase.
        cs2 : float
            Speed of sound squared in the symmetric phase.
        psiN : float
            Ratio of enthalpies at nucleation.

        Returns
        -------
        float
            The strength of the phase transition at the wall.
        """
        nu, mu = 1+1/ cb2 ,1+1/ cs2
        vm = min(np.sqrt(cb2), vw)
        vp_max = min(cs2 / vw, vw)
        al_min = max ((vm - vp_max) *(cb2 - vm * vp_max) /(3* cb2 * vm *(1 - vp_max**2)),
                      (mu - nu) /(3* mu))
        al_max = 1/3
        branch = -1
        if self.eqWall(al_min, alN, vm, nu, mu, psiN) * \
           self.eqWall(al_max, alN, vm, nu, mu, psiN) > 0:
            branch = 1

        sol = root_scalar(self.eqWall, (alN, vm, nu, mu, psiN, branch),
                          bracket=(al_min, al_max), rtol=1e-10, xtol=1e-10)

        if not sol.converged and self.verbose:
            print("WARNING: desired precision not reached in 'solve_alpha'")
        return sol.root

    def dfdv(self, v : float, X : np.ndarray, cs2 : float) -> list[float]:
        """
        Compute the derivatives for the hydrodynamic equations.

        Parameters
        ----------
        v : float
            Velocity of the fluid.
        X : np.ndarray
            Array containing the variables [xi, w].
        cs2 : float
            Speed of sound squared.

        Returns
        -------
        list[float]
            The derivatives [dXidv, dwdv].
        """
        xi, w = X
        with np.errstate(divide='ignore', invalid='ignore'):
            mu_xiv = (xi - v) / (1 - xi * v)
            dxidv = xi * (1 - v * xi) * (mu_xiv**2 / cs2 - 1) / (2 * v * (1 - v**2))
            dwdv = w * (1 + 1 / cs2) * mu_xiv / (1 - v**2)
        return [dxidv, dwdv]

    def integrate_plasma(self, v0, vw, w0, c2, shock_wave=True):
        """
        Integrate the hydrodynamic equations for the plasma profile.

        Parameters
        ----------
        v0 : float
            Initial velocity of the fluid.
        vw : float
            Velocity of the fluid at the wall.
        w0 : float
            Initial value of w.
        c2 : float
            Speed of sound squared.
        shock_wave : bool
            Whether to include shock wave effects.

        Returns
        ----------
        OdeResult
            The result of the integration containing the plasma profile.
        """
        def event (v ,X, cs2):
            xi, w = X
            return xi*(xi - v) / (1 - xi*v) - cs2

        event.terminal = True
        sol = None
        if shock_wave :
            sol = solve_ivp(self.dfdv, (v0 ,1e-20), [ vw, w0] ,
                            events= event, args=(c2,), rtol=1e-10, atol=1e-10)
        else:
            sol = solve_ivp(self.dfdv,(v0 ,1e-20), [vw, w0] ,
                            args=(c2,), rtol=1e-10, atol=1e-10)
        if not sol.success and self.verbose:
            print(" WARNING : desired precision not reached in 'integrate_plasma'")

        return sol

    def shooting(self, vw : float, alN : float, cb2 : float, cs2 : float, psiN : float) -> float:
        """ Shooting method to find the wall velocity consistent with matching conditions.
        
        Parameters
        ----------
        vw : float
            Velocity of the fluid at the wall.
        alN : float
            Strength of the phase transition at nucleation.
        cb2 : float
            Speed of sound squared in the broken phase.
        cs2 : float
            Speed of sound squared in the symmetric phase.
        psiN : float
            Ratio of enthalpies at nucleation.
        
        Returns
        -------
        float
            The value of the shooting function.
        """
        nu, mu = 1+1/cb2 ,1+1/ cs2
        vm = min(np.sqrt(cb2), vw)
        al = self.solve_alpha(vw, alN, cb2, cs2, psiN)
        vp = self.get_vp(vm, al, cb2)
        wp = self.w_from_alpha(al, alN, nu, mu)
        sol = self.integrate_plasma ((vw - vp) /(1 - vw * vp), vw, wp, cs2)
        vp_sw = sol.y[0, -1]
        vm_sw = (vp_sw - sol.t[-1]) /(1 - vp_sw * sol.t[-1])
        wm_sw = sol.y[1, -1]
        return vp_sw / vm_sw - ((mu -1) * wm_sw +1) /((mu -1) + wm_sw)

    def max_al(self, cb2 : float, cs2 : float, psiN : float,
               upper_limit : float =1) -> float:
        """Return the maximal nucleation-strength compatible with
        subsonic solutions.
        
        Parameters
        ----------
        cb2 : float
            Speed of sound squared in the broken phase.
        cs2 : float
            Speed of sound squared in the symmetric phase.
        psiN : float
            Ratio of enthalpies at nucleation.
        upper_limit : float
            Upper limit for the search of maximal alpha.
        
        Returns
        -------
        float
            The maximal nucleation-strength compatible with subsonic solutions.
        
        """
        nu, mu = 1+1/ cb2 ,1+1/ cs2
        vm = np.sqrt(cb2)

        def func(alN):
            vw = self.find_vJ(alN, cb2)
            vp = cs2 / vw
            ga2p, ga2m = 1/(1 - vp**2) ,1/(1 - vm**2)
            wp = (vp + vw - vw * mu) /(vp + vw - vp * mu)
            psi = psiN * wp**(nu / mu -1)
            al = (mu - nu) /(3* mu) +(alN -(mu - nu) /(3* mu)) / wp
            return vp * vm * al /(1 -(nu -1) * vp * vm) -(
                1 -3* al -(ga2p / ga2m)**(nu /2) * psi) /(3*nu)

        if func(upper_limit) < 0:
            return upper_limit
        sol = root_scalar(func, bracket =((1 - psiN) /3, upper_limit),
                          rtol =1e-10, xtol =1e-10)
        return sol.root

    def find_vw(self, alN : float, cb2 : float, cs2 : float,
                psiN : float) -> float:
        """Solve for the wall velocity consistent with matching conditions.
        
        Parameters
        ----------
        alN : float
            Strength of the phase transition at nucleation.
        cb2 : float
            Speed of sound squared in the broken phase.
        cs2 : float
            Speed of sound squared in the symmetric phase.
        psiN : float
            Ratio of enthalpies at nucleation.
            
        Returns
        -------
        float
            The wall velocity consistent with matching conditions.

        """
        nu, mu = 1+1/ cb2 ,1+1/ cs2
        vJ = self.find_vJ(alN, cb2)
        if alN < (1 - psiN) /3 or alN <=(mu - nu) /(3* mu):
            if self.verbose:
                print("alpha_N is too small to find a solution for the wall velocity, set it to 0.")
            return 0
        if alN > self.max_al(cb2, cs2, psiN ,100) or self.shooting(vJ, alN, cb2, cs2, psiN) < 0:
            if self.verbose:
                print("alpha_N too large to find a solution for the wall velocity, set it to 1.")
            return 1
        sol = root_scalar(self.shooting, (alN, cb2, cs2, psiN) ,
                          bracket =[1e-3, vJ], rtol =1e-10, xtol =1e-10)
        return sol.root

    def detonation(self, alN : float, cb2 : float, psiN : float) -> float:
        """Return the detonation wall velocity if a solution exists.
        
        Parameters
        ----------
        alN : float
            Strength of the phase transition at nucleation.
        cb2 : float
            Speed of sound squared in the broken phase.
        psiN : float
            Ratio of enthalpies at nucleation.
            
        Returns
        -------
        float
            The detonation wall velocity if a solution exists, otherwise 0.
        """
        nu = 1+1/cb2
        vJ = self.find_vJ(alN, cb2)

        def matching_eq(vw):
            A = vw**2+ cb2 *(1 -3* alN *(1 - vw**2))
            vm =(A + np.sqrt(A**2 -4* vw**2* cb2)) /(2* vw)
            ga2w, ga2m = 1/(1 - vw**2) ,1/(1 - vm**2)
            return vw * vm * alN /(1 -(nu -1) * vw * vm) \
                -(1 -3* alN -(ga2w / ga2m)**(nu /2) * psiN)/(3* nu)

        if matching_eq(vJ +1e-10) * matching_eq (1 -1e-10) > 0:
            if self.verbose:
                print('No detonation solution')
            return 0

        sol = root_scalar(matching_eq, bracket =(vJ +1e-10 ,1 -1e-10),
                          rtol =1e-10, xtol =1e-10)
        return sol.root

    def find_kappa(self, alN : float, cb2 : float, cs2 : float,
                   psiN : float, vw : float = None) -> float:
        """Compute the hydrodynamic efficiency factor for given plasma parameters.

        Parameters
        ----------
        alN : float
            Strength of the phase transition at nucleation.
        cb2 : float
            Speed of sound squared in the broken phase.
        cs2 : float
            Speed of sound squared in the symmetric phase.
        psiN : float
            Ratio of enthalpies at nucleation.
        vw : float, optional
            Wall velocity. If not provided, it will be computed.

        Returns
        -------
        float
            The hydrodynamic efficiency factor.
        """
        if vw is None:
            vw = self.find_vw(alN, cb2, cs2, psiN)
        nu, mu = 1 + 1/cb2, 1 + 1/cs2
        kappa, wp, vm, vp = 0, 1 ,0 ,0
        if vw < 1:
            vm = min(np.sqrt(cb2), vw)
            al = self.solve_alpha(vw, alN, cb2, cs2, psiN)
            vp = self.get_vp(vm, al, cb2)
            wp = self.w_from_alpha(al, alN, nu, mu)
            sol = self.integrate_plasma ((vw - vp) /(1 - vw * vp) ,vw, wp, cs2)
            v, xi, w = sol.t, sol.y[0], sol.y [1]
            kappa += 4 * simpson((xi * v)**2* w /(1 - v**2), xi) /(vw**3* alN)
        if vw**2 > cb2:
            w0 = psiN * wp**(nu / mu) * ((1 - vm**2)/(1 - vp**2))**(nu/2) \
                if vw < 1 else 1+6*alN/(nu-2)
            v0 = (vw - vm) / (1 - vw * vm) if vw < 1 else 3 * alN / (nu-2+3*alN)
            sol = self.integrate_plasma(v0, vw, w0, cb2, False)
            v, xi, w = np.flip(sol.t), np.flip(sol.y [0]), np.flip(sol.y [1])
            mask = np.append(xi[1:] > xi [: -1], True)
            kappa += 4 * simpson(((xi * v)**2* w /(1 - v**2))[mask], xi[mask]) \
                /(vw**3* alN)
        return kappa


def calc_kappas(alphaN: float, alpha_inf: float, alpha_eq: float,
                vw: float, cs: float, Rsep: float, R0: float, config) -> tuple:
    """
    Calculate the kappa parameters kappa_phi, kappa_sw and kappa_turb.
    These parameters quantify the amount of energy that is available
    for gravitational wave production in the broken phase due to
    bubble collisions, sound waves and turbulence.

    Parameters
    ----------
    alphaN : float
        The strength of the phase transition at nucleation.
    alpha_inf : float
        The critical strength of the phase transition at which bubbles
        would run away if there was only leading order friction.
    alpha_eq : float
        The critical strength of the phase transition at which
        the next-to-leading order friction is compensated by a
        high bubble wall velocity.
    vw : float
        The bubble wall velocity.
    cs : float
        The speed of sound in the broken phase.
    Rsep : float
        The mean bubble separation at the time of percolation.
    R0 : float
        The critical bubble radius at which the phase transition
        starts at nucleation.
    config : object
        The configuration object containing for instance the
        epsilon parameter determining the importance of
        turbulence in the broken phase.

    Returns
    ----------
    kappa_bw : float
        The kappa parameter for bubble wall collisions.
    kappa_sw : float
        The kappa parameter for sound waves.
    kappa_turb : float
        The kappa parameter for turbulence.
    """
    # unknown parameter epsilon is to represent the fraction of overall kinetic energy
    # in bulk motion that is converted to MHD turbulence, see arXiv:2403.03723 eq. (2.13)
    eps = config.epsilon_turbulence

    # Step 1:
    # Computation of kappa_phi
    # based on 1903.09642 by John Ellis, Marek 
    # Lewicki, José Miguel No & Ville Vaskonen
    gamma_eq = np.inf if alpha_eq == 0 else (alphaN - alpha_inf) / alpha_eq
    gamma_star = 2 * Rsep / (3 * R0)

    def kappa_col(gamma_star, gamma_eq, alpha_inf, alpha):
        if alpha < alpha_inf:
            # If the leading-order friction is so large that the bubbles would not even reach
            # relativistic terminal velocity, we set kappa_BW = 0.
            return 0
        if gamma_star > gamma_eq:
            return (gamma_eq / gamma_star) * (1 - (alpha_inf / alpha) * (gamma_eq / gamma_star)**2)
        else:
            return 1 - alpha_inf / alpha

    kappa_BW = 0
    if config.bw_collisions == "off":
        kappa_BW = 0
    elif config.bw_collisions == "full":
        kappa_BW = 1
    elif config.bw_collisions == "NLO":
        kappa_BW = kappa_col(gamma_star, gamma_eq, alpha_inf, alphaN)
    else:
        raise ValueError("Wrong config input: bw_collisions must be 'off', 'full' or 'NLO'.")

    # Step 2:
    # Computation of kappa_sw (assuming no kappa_phi)
    # based on  1004.4187 by Jose R. Espinosa, Thomas Konstandin,
    # Jose M. No & Geraldine Servant

    alpha_eff = alphaN * (1 - kappa_BW)
    kappa_SW = (1 - kappa_BW) * kappa_sw(alpha_eff, vw, cs)
    kappa_TURB = eps * kappa_SW
    return kappa_BW, kappa_SW, kappa_TURB
