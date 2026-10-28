"""
A module for finding instantons between vacua in multiple field dimensions.

The basic strategy is an iterative process:

    1. Make an ansatz for the path along which the field will travel.
    2. Split up the equations of motion into components that are parallel and
       perpendicular to the direction of travel along the path.
    3. The direction of motion parallel to the path reduces to a
       one-dimensional equation of motion, which can be solved using the
       overshoot / undershoot techniques in :mod:`.tunneling1D`. Solve it.
    4. Treating the motion of the field as a classical particle moving in an
       inverted potential, calculate the normal forces that would need to act
       on the particle to keep it on the path. If this forces are (close enough
       to) zero, the ansatz was correctly. Otherwise iteratively deform the path
       in the direction of the normal forces, stopping when the forces go to
       zero.
    5. Loop back to step 3 until no further deformation is necessary.

The classes :class:`Deformation_Spline` and :class:`Deformation_Points` will
perform step 3, while :func:`fullTunneling` will run the entire loop.

For more explicit details, see the original paper
`Comput. Phys. Commun. 183 (2012)`_ [`arXiv:1109.4189`_].

.. _`Comput. Phys. Commun. 183 (2012)`:
    http://dx.doi.org/10.1016/j.cpc.2012.04.004

.. _`arXiv:1109.4189`: http://arxiv.org/abs/1109.4189

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

from typing import Callable
import numpy as np
from dataclasses import dataclass
from scipy import optimize, interpolate, integrate
from collections import namedtuple
from . import tunneling1D
from . import helper_functions
from . import errors

class Deformation_Spline(object):
    """
    Deform a path in the presence of a potential such that the normal forces
    along the path are zero.

    This class fits a spline to the points, and does the actual deformation
    on the spline rather than on the points themselves. This make the path
    somewhat smoother than it would otherwise be (which is generally desirable),
    but it does make it difficult to resolve sharp turns in the path.

    Parameters
    ----------
    phi : array_like
        The list of points that constitutes the original path. Should have
        shape ``(n_points, n_dimensions)``.
    dphidr : array_like
        The 'speed' along the path at the initial points. This does not change
        as the path deforms. Should have shape ``(n_points,)``. Gets saved into
        the attribute `v2` as ``v2 = dphidr[:,np.newaxis]**2``.
    dV : callable
        The potential gradient as a function of phi. The output shape should be
        the same as the input shape, which will be ``(..., n_dimensions)``.
    nb : int, optional
        Number of basis splines to use.
    kb : int, optional
        Order of basis splines.
    v2min : float, optional
        The smallest the square of dphidr is allowed to be, relative
        to the characteristic force exterted by F_ext. Note that the
        self-correcting nature of the deformation goes away when dphidr=0.
    fix_start, fix_end : bool, optional
        If True, the force on the first/last point along the path is set to
        zero, so the point will not change in the deformation step.
    save_all_steps : bool, optional
        If True, each step gets saved into ``self.phi_list`` and
        ``self.F_list``.

    Attributes
    ----------
    phi : array_like
        Set during initialization, and then rewritten at each step.
    num_steps : int
        Total number of steps taken.
    """
    """
    Additional (private) attributes
    -------------------------------
    _L : float
        Total length of the path, set during initialization.
    _t : array_like
        Array from (0,1] marking the locations of each point.
    _X, _dX, _d2X : array_like
        Spline basis functions and their derivatives evaluated at `_t`. Set
        during initialization.
    _beta : array_like
        The spline coefficients for each dimension. Recalculated each step.
    _F_prev, _phi_prev : array_like
        The normal force and the path points at the last step.
    """
    def __init__(self, phi: np.ndarray, dphidr: np.ndarray, dV: callable,
                 nb: int=10, kb: int=3, v2min: float=0.0,
                 fix_start: bool=False, fix_end: bool=False,
                 save_all_steps: bool=False):
        """Pre-compute spline basis functions and store the deformation state."""
        # First step: convert phi to a set of path lengths.
        phi = np.asanyarray(phi)
        dphi = phi[1:]-phi[:-1]
        dL = np.sqrt(np.sum(dphi*dphi,axis=-1))
        y = np.cumsum(dL)
        self._L = y[-1]
        self._t = np.append(0,y)/self._L
        self._t[0] = 1e-100  # Without this, the first data point isn't in
                            # any bin (this matters for dX).

        # Create the starting spline:
        # make the knots and then the spline matrices at each point t
        t0 = np.append(np.append([0.]*(kb-1), np.linspace(0,1,nb+3-kb)),
                       [1.]*(kb-1))
        self._X,self._dX,self._d2X = helper_functions.Nbspld2(t0, self._t, kb)
        self._t = self._t[:,np.newaxis]  # Shape (n, 1)
        # subtract off the linear component.
        phi0, phi1 = phi[:1], phi[-1:]  # These are shape (1,N)
        phi_lin = phi0 + (phi1-phi0)*self._t
        self._beta, residues, rank, s = np.linalg.lstsq(self._X, phi - phi_lin, rcond=None)

        # save the points for future use.
        self.phi = np.asanyarray(phi)  # shape (n,N)
        self.v2 = np.asanyarray(dphidr)[:,np.newaxis]**2  # shape (n,1)
        self.dV = dV
        self.F_list = []
        self.phi_list = []
        self._phi_prev = self._F_prev = None
        self.save_all_steps = save_all_steps
        self.fix_start, self.fix_end = fix_start, fix_end
        self.num_steps = 0

        # ensure that v2 isn't too small:
        v2 = dphidr**2
        v2min *= np.max(np.sum(dV(self.phi)**2, -1)**.5*self._L/nb)
        v2[v2 < v2min] = v2min
        self.v2 = v2[:,np.newaxis]

    _forces_rval = namedtuple("forces_rval", "F_norm dV")
    def forces(self):
        """
        Calculate the normal force and potential gradient on the path.

        Returns
        -------
        F_norm, dV : array_like
        """
        X, dX, d2X = self._X, self._dX, self._d2X
        beta = self._beta
        # First find phi, dphi, and d2phi. Note that dphi needs to get a
        # linear component added in, while d2phi does not.
        phi = self.phi
        dphi = np.sum(beta[np.newaxis,:,:]*dX[:,:,np.newaxis], axis=1) \
            + (self.phi[-1]-self.phi[1])[np.newaxis,:]
        d2phi = np.sum(beta[np.newaxis,:,:]*d2X[:,:,np.newaxis], axis=1)
        # Compute dphi/ds, where s is the path length instead of the path
        # parameter t. This is just the direction along the path.
        dphi_sq = np.sum(dphi*dphi, axis=-1)[:,np.newaxis]
        dphids = dphi/np.sqrt(dphi_sq)
        # Then find the acceleration along the path, i.e. d2phi/ds2:
        d2phids2 = (d2phi - dphi * np.sum(dphi*d2phi, axis=-1)[:,np.newaxis] /
                    dphi_sq)/dphi_sq
        # Now we have the path at the points t, as well as its derivatives with
        # respect to its path length. We still need the normal force on the path.
        dV = self.dV(phi)
        dV_perp = dV - np.sum(dV*dphids, axis=-1)[:,np.newaxis]*dphids
        F_norm = d2phids2 * self.v2 - dV_perp
        if (self.fix_start):
            F_norm[0] = 0.0
        if (self.fix_end):
            F_norm[-1] = 0.0
        return self._forces_rval(F_norm, dV)

    _step_rval = namedtuple("step_rval", "stepsize step_reversed fRatio")
    def step(self, lastStep: float, maxstep: float=.1, minstep: float=1e-4,
             reverseCheck: float=.15, stepIncrease: float=1.5,
             stepDecrease: float=5., checkAfterFit: bool=True,
             verbose: bool=False):
        """
        Deform the path one step.

        Each point is pushed in the direction of the normal force - the force
        that the path exerts on a classical particle moving with speed `dphidr`
        in a potential with gradient `dV` such that the particle stays on the
        path. A stepsize of 1 corresponds to moving the path an amount
        ``L*N/(dV_max)``, where `L` is the length of the (original) path,
        `N` is the normal force, and `dV_max` is the maximum force exerted by
        the potential along the path.

        Parameters
        ----------
        lastStep : float
            Size of the last step.
        maxstep, minstep : float, optional
        reverseCheck : float, optional
            Percentage of points for which the force can reverse direcitons
            (relative to the last step) before the stepsize is decreased.
            If ``reverseCheck >= 1``, the stepsize is kept at `lastStep`.
        stepIncrease, stepDecrease : float, optional
            The amount to increase or decrease stepsize over the last step.
            Both should be bigger than 1.
        checkAfterFit : bool, optional
            If True, the convergence test is performed after the points are fit
            to a spline. If False, it's done beforehand.
        verbose : bool, optional
            If True, output is printed at each step.

        Returns
        -------
        stepsize : float
            The stepsize used for this step.
        step_reversed : bool
            True if this step was reversed, otherwise False
        fRatio : float
            The ratio of the maximum normal force to the maximum potential
            gradient. When the path is a perfect fit, this should go to zero. If
            ``checkAfterFit == True``, the normal force in this ratio is defined
            by the change in phi this step *after* being fit to a spline. Note
            that if the spline does a poor job of fitting the points after the
            deformation in this step (which might be the case if there are not
            enough basis functions), and if ``checkAfterFit == False``, this
            ratio can be non-zero or large even if there is no change in ``phi``.

        Notes
        -----
        In prior versions of this function (CosmoTransitions v1.0.2 and
        earlier), the start and end points of the
        path were effectively held fixed during the main deformation. This was
        because the line ``phi_lin = phi[:1] + ...`` was calculated *before* the
        line ``phi = phi+F*stepsize``. Since the spline basis functions are
        zero at the start and end points (the spline is added on top of the
        straight line between the end points), when the points were later taken
        from the spline the end points wouldn't move. This was by design, since
        for thin-walled bubbles the endpoints should stay fixed at the two
        vacua. However, this caused problems for thick-walled bubbles where the
        end points should move.

        To get around this, prior versions added an extra block of code to move
        the end points before the main deformation. However, this was
        unnecessarily complicated and led to error-prone code. In this version,
        the end points are always allowed to move if the force `F` is non-zero.
        In the thin-walled case, the force should be almost exactly zero at
        the end points anyways (there is zero potential gradient and `dphidr` is
        zero), so they should stay fixed on their own.
        """
        # Find out the direction of the deformation.
        F, dV = self.forces()
        F_max = np.max(np.sqrt(np.sum(F*F,-1)))
        dV_max = np.max(np.sqrt(np.sum(dV*dV,-1)))
        fRatio1 = F_max / dV_max
        # Rescale the normal force so that it's relative to L:
        F *= self._L / dV_max

        # Now, see how big the stepsize should be
        stepsize = lastStep
        phi = self.phi
        assert(maxstep > minstep)
        step_reversed = False
        if reverseCheck < 1 and self._F_prev is not None:
            FdotFlast = np.sum(F*self._F_prev, axis=1)
            if np.sum(FdotFlast < 0) > len(FdotFlast)*reverseCheck:
                # we want to reverse the last step
                if stepsize > minstep:
                    step_reversed = True
                    phi = self._phi_prev
                    F = self._F_prev
                    if verbose: print("step reversed")
                    stepsize = lastStep/stepDecrease
            else:
                # No (large number of) indices reversed, just do a regular
                # step. Increase the stepsize a bit over the last one.
                stepsize = lastStep * stepIncrease
        if stepsize > maxstep: stepsize = maxstep
        if stepsize < minstep: stepsize = minstep

        # Save the state before the step
        self._phi_prev = phi
        self._F_prev = F
        if self.save_all_steps:
            self.phi_list.append(phi)
            self.F_list.append(F)

        # Now make the step. It's important to not use += so we do not mutate
        # the value stored in self.phi_list.
        phi = phi+F*stepsize

        # fit to the spline
        phi_lin = phi[:1] + (phi[-1:]-phi[:1])*self._t
        phi -= phi_lin
        self._beta, residues, rank, s = np.linalg.lstsq(self._X, phi, rcond=None)
        phi = np.sum(self._beta[np.newaxis,:,:]*self._X[:,:,np.newaxis], axis=1)
        phi += phi_lin
        self.phi = phi

        Ffit = (phi-self._phi_prev)/stepsize
        fRatio2 = np.max(np.sqrt(np.sum(Ffit*Ffit,-1)))/self._L

        if verbose:
            print("step: %i; stepsize: %0.2e; fRatio1 %0.2e; fRatio2: %0.2e"
                  % (self.num_steps, stepsize, fRatio1, fRatio2))

        fRatio = fRatio2 if checkAfterFit else fRatio1
        return self._step_rval(stepsize, step_reversed, fRatio)

    def deformPath(self, startstep : float=2e-3, fRatioConv: float=.02,
                   converge_0: float=5., fRatioIncrease: float=5.,
                   maxiter: int=500, verbose: bool=False, callback=None,
                   step_params={}) -> bool:
        """
        Deform the path many individual steps, stopping either when the
        convergence criterium is reached, when the maximum number of iterations
        is reached, or when the path appears to be running away from
        convergence.

        Parameters
        ----------
        startstep : float, optional
            Starting stepsize used in :func:`step`.
        fRatioConv : float, optional
            The routine will stop when the maximum normal force on the path
            divided by the maximum potential gradient is less than this.
        converge_0 : float, optional
            On the first step, use a different convergence criterion. Check if
            ``fRatio < convergence_0 * fRatioConv``.
        fRatioIncrease :float, optional
            The maximum fractional amount that fRatio can increase before
            raising an error.
        maxiter : int, optional
            Maximum number of steps to take (ignoring reversed steps).
        verbose : bool, optional
            If True, print the ending condition.
        callback : callable, optional
            Called after each step. Should accept an instance of this class as a
            parameter, and return False if deformation should stop.
        step_params : dict, optional
            Parameters to pass to :func:`step`.

        Returns
        -------
        deformation_converged : bool
            True if the routine stopped because it converged (as determined by
            `fRatioConv`), False otherwise.
        """
        minfRatio = np.inf
        minfRatio_index = 0
        minfRatio_beta = None
        minfRatio_phi = None
        stepsize = startstep
        deformation_converged = False
        while True:
            self.num_steps += 1
            stepsize, step_reversed, fRatio = self.step(stepsize, **step_params)
            if callback is not None and not callback(self):
                break
            minfRatio = min(minfRatio, fRatio)
            if fRatio < fRatioConv or (self.num_steps == 1
                                       and fRatio < converge_0*fRatioConv):
                if verbose:
                    print("Path deformation converged. " + "%i steps. fRatio = %0.5e" % (self.num_steps,fRatio))
                deformation_converged = True
                break
            if minfRatio == fRatio:
                minfRatio_beta = self._beta
                minfRatio_index = self.num_steps
                minfRatio_phi = self.phi
            if fRatio > fRatioIncrease*minfRatio and not step_reversed:
                self._beta = minfRatio_beta
                self.phi = minfRatio_phi
                self.phi_list = self.phi_list[:minfRatio_index]
                self.F_list = self.F_list[:minfRatio_index]
                raise errors.DeformationError()
            if self.num_steps >= maxiter:
                if verbose:
                    print("Maximum number of deformation iterations reached.")
                break
        return deformation_converged


class Deformation_Points(object):
    """
    Deform a path in the presence of a potential such that the normal forces along the path are
    zero.

    See :ref:`note-path-deformation-deformation-points` for the reasoning and the measurements
    behind this.

    Parameters
    ----------
    phi : array_like
        The list of points that constitutes the original path. Should have
        shape ``(n_points, n_dimensions)``.
    dphidr : array_like
        The 'speed' along the path at the initial points. This does not change
        as the path deforms. Should have shape ``(n_points,)``. Gets saved into
        the attribute ``self.v2`` as ``v2 = dphidr[:,np.newaxis]**2``.
    dV : callable
        The potential gradient as a function of phi. The output shape should be
        the same as the input shape, which will be ``(..., n_dimensions)``.
    fix_start, fix_end : bool, optional
        If True, the force on the first/last point along the path is set to
        zero, so the point will not change in the deformation step.
    save_all_steps : bool, optional
        If True, each step gets saved into ``self.phi_list`` and
        ``self.F_list``.

    Attributes
    ----------
    phi : array_like
        Set during initialization, and then rewritten at each step.
    num_steps : int
        Total number of steps taken.
    """
    def __init__(self, phi: np.ndarray, dphidr: np.ndarray,
                 dV: Callable[[np.ndarray], np.ndarray],
                 fix_start=False, fix_end=False, save_all_steps=False):
        """Store the points and derivative data required for
        point-wise deformation."""
        self.phi = np.asanyarray(phi)  # shape (n,N)
        self.v2 = np.asanyarray(dphidr)[:,np.newaxis]**2  # shape (n,1)
        self.dV = dV
        self.F_list = []
        self.phi_list = []
        self.save_all_steps = save_all_steps
        self.fix_start, self.fix_end = fix_start, fix_end
        self.num_steps = 0

    _forces_rval = namedtuple("forces_rval", "F_norm dV")
    def forces(self, phi : np.ndarray=None):
        """
        Calculate the normal force and potential gradient on the path.

        Returns
        -------
        F_norm, dV : array_like
        """
        if phi is None: phi = self.phi
        # Let `t` be some variable that parametrizes the points such that
        # t_i = i. Calculate the derivs of phi w/ respect to t.
        dphi = helper_functions.deriv14_const_dx(phi.T).T
        d2phi = helper_functions.deriv23_const_dx(phi.T).T
        # Let `x` be some variable that parametrizes the path such that
        # |dphi/dx| = 1. Calculate the derivs.
        dphi_abssq = np.sum(dphi*dphi, axis=-1)[:,np.newaxis]
        dphi /= np.sqrt(dphi_abssq)  # This is now dphi/dx
        d2phi /= dphi_abssq  # = d2phi/dx2 + (dphi/dx)(d2phi/dt2)/(dphi/dt)^2
        d2phi -= np.sum(d2phi*dphi, axis=-1)[:,np.newaxis] * dphi  # = d2phi/dx2
        # Calculate the total force.
        dV = self.dV(phi)
        dV_perp = dV - np.sum(dV*dphi, axis=-1)[:,np.newaxis] * dphi
        F_norm = d2phi*self.v2 - dV_perp
        if (self.fix_start):
            F_norm[0] = 0.0
        if (self.fix_end):
            F_norm[-1] = 0.0
        return self._forces_rval(F_norm, dV)

    _step_rval = namedtuple("step_rval", "stepsize fRatio")
    def step(self, stepsize: float, minstep: float,
             diff_check: float=0.1, step_decrease: float=2.):
        """
        Take two half-steps in the direction of the normal force.

        Parameters
        ----------
        stepsize : float
            Determines change in ``phi``: ``phi += F_norm*stepsize``.
        minstep : float
            The smallest the stepsize is allowed to be.
        diff_check : float, optional
            The stepsize is chosen such that difference between the forces at
            beginning of the step and halfway through the step is small
            compared to the force itself: ``max(F2-F1) < diff_check * max(F1)``,
            where ``max`` here really means the maximum absolute value of the
            force in each direction.
        step_decrease : float, optional
            Amount by which to decrease the stepsize if the step is too big.

        Returns
        -------
        stepsize : float
            The stepsize used for this step.
        fRatio : float
            The ratio of the maximum normal force to the maximum potential
            gradient. When the path is a perfect fit, this should go to zero.
        """
        F1, dV = self.forces()
        F_max = np.max(np.sqrt(np.sum(F1*F1,-1)))
        dV_max = np.max(np.sqrt(np.sum(dV*dV,-1)))
        fRatio = F_max / dV_max

        if self.save_all_steps:
            self.phi_list.append(self.phi)
            self.F_list.append(F1)

        while True:
            # Take one full step
            #  phi1 = self.phi + F*stepsize
            # Take two half steps
            phi2 = self.phi + F1*(stepsize*0.5)
            F2 = self.forces(phi2)[0]
            if stepsize <= minstep:
                stepsize = minstep
                break
          #  phi2 += F2*(stepsize*0.5)
            DF_max = np.max(np.abs(F2-F1), axis=0)
            F_max = np.max(np.abs(F1), axis=0)
            if (DF_max < diff_check*F_max).all():
                break
            stepsize /= step_decrease
        self.phi = phi2 + F2*(stepsize*0.5)

        return self._step_rval(stepsize, fRatio)

    def deformPath(self, startstep=.1, minstep=1e-6, step_increase=1.5,
                   fRatioConv=.02, converge_0=5., fRatioIncrease=20.,
                   maxiter=500, verbose=0, callback=None, step_params={}):
        """
        Deform the path many individual steps, stopping either when the
        convergence criterium is reached, when the maximum number of iterations
        is reached, or when the path appears to be running away from
        convergence.

        Parameters
        ----------
        startstep, minstep : float, optional
            Starting and minimum stepsizes passed to :func:`step`, rescaled by
            ``sum(|phi[0]-phi[-1]|) / (num_points * max(dV))``, the first and last points of
            the path rather than the first two. ``minstep`` is the lower bound on the
            stepsize; :func:`step` takes no upper bound, and only ever divides the stepsize by
            ``step_decrease``, while :func:`deformPath` multiplies it by ``step_increase``
            after each step.
        fRatioConv : float, optional
            The routine will stop when the maximum normal force on the path
            divided by the maximum potential gradient is less than this.
        converge_0 : float, optional
            On the first step, use a different convergence criterion. Check if
            ``fRatio < convergence_0 * fRatioConv``.
        fRatioIncrease :float, optional
            The maximum fractional amount that fRatio can increase before
            raising an error.
        maxiter : int, optional
            Maximum number of steps to take (ignoring reversed steps).
        verbose : int, optional
            If ``verbose >= 1``, print the ending condition.
            If ``verbose >= 2``, print `fRatio` and `stepsize` at each step.
        callback : callable, optional
            Called after each step. Should accept an instance of this class as a
            parameter, and return False if deformation should stop.
        step_params : dict, optional
            Parameters to pass to :func:`step`.

        Returns
        -------
        deformation_converged : bool
            True if the routine stopped because it converged (as determined by
            `fRatioConv`), False otherwise.
        """
        minfRatio = np.inf
        Delta_phi = np.sum(np.sqrt((self.phi[0]-self.phi[-1])**2))
        dV_max = np.max(np.sum(self.dV(self.phi)**2, axis=-1))**0.5
        step_scale = Delta_phi / (len(self.phi) * dV_max)
        stepsize = startstep * step_scale
        minstep *= step_scale
        deformation_converged = False
        while True:
            self.num_steps += 1
            stepsize, fRatio = self.step(stepsize, minstep, **step_params)
            if verbose >= 2:
                print("step: %i; stepsize: %0.2e; fRatio: %0.2e"
                      % (self.num_steps, stepsize, fRatio))
            stepsize *= step_increase
            if callback is not None and not callback(self):
                break
            minfRatio = min(minfRatio, fRatio)
            if fRatio < fRatioConv or (self.num_steps == 1
                                       and fRatio < converge_0*fRatioConv):
                if verbose >= 1:
                    print("Path deformation converged." +
                          "%i steps. fRatio = %0.5e" % (self.num_steps,fRatio))
                deformation_converged = True
                break
            if minfRatio == fRatio:
                minfRatio_index = self.num_steps
                minfRatio_phi = self.phi
            if fRatio > fRatioIncrease*minfRatio:
                self.phi = minfRatio_phi
                self.phi_list = self.phi_list[:minfRatio_index]
                self.F_list = self.F_list[:minfRatio_index]
                raise errors.DeformationError()
            if self.num_steps >= maxiter:
                if verbose: print("Maximum number of iterations reached.")
                break
        return deformation_converged


_extrapolatePhi_rtype = namedtuple("extrapolatePhi_rval", "phi s L")
def _extrapolatePhi(phi0, V=None, tails=0.2):
    """
    Returns a list of points along the path, going linearly
    beyond the path to include the nearest minima.

    Parameters
    ----------
    phi0 : array_like
        The (multi-dimensional) path to extend.
    V : callable or None
        The potential to minimize, or None if the path should be extended a
        fixed amount beyond its ends.
    tails : float
        The amount relative to the path length to extrapolate beyond the end of
        the path (if V is None) or beyond the minima (if V is not None).

    Returns
    -------
    phi : array_like
        The extended list of points. The spacing between points in the extended
        regions should be approximately the same as the spacing between the
        input points.
    s : array_like
        The distance along the path (starting at ``phi0[0]``).
    L : float
        Total length of the path excluding tails.
    """
    phi1 = phi = phi0
    dphi = np.append(0,np.sum((phi1[1:]-phi1[:-1])**2,1)**.5)
    s1 = np.cumsum(dphi)
    L = s1[-1]
    npoints = phi1.shape[0]

    phi_hat0 = (phi[1]-phi[0])/np.sum((phi[1]-phi[0])**2)**.5
    if V is None:
        s0min = 0.0
    else:
        V0 = lambda x: V(phi[0] + phi_hat0*x*L)
        s0min = optimize.fmin(V0, 0.0, disp=0, xtol=1e-5)[0]*L
    if s0min > 0: s0min = 0.0
    s0 = np.linspace(s0min - L*tails, 0.0, npoints*tails)[:-1]
    phi0 = phi[0] + phi_hat0*s0[:,np.newaxis]

    phi_hat2 = (phi[-1]-phi[-2])/np.sum((phi[-1]-phi[-2])**2)**.5
    if V is None:
        s2min = 0.0
    else:
        V2 = lambda x: V(phi[-1] + phi_hat2*(x-1)*L)
        s2min = optimize.fmin(V2, 1, disp=0, xtol=1e-5)[0]*L
    if s2min < L: s2min = L
    s2 = np.linspace(L, s2min + L*tails, npoints*tails)[1:]
    phi2 = phi[-1] + phi_hat2*(s2[:,np.newaxis]-L)

    phi = np.append(phi0, np.append(phi1, phi2, 0), 0)
    s = np.append(s0, np.append(s1, s2))

    return _extrapolatePhi_rtype(phi, s, L)


def _pathDeriv(phi):
    """Calculates to 4th order if len(phi) >= 5, otherwise 1st/2nd order."""
    if len(phi) >= 5:
        dphi = helper_functions.deriv14_const_dx(phi.T).T
    elif len(phi) > 2:
        dphi = np.empty_like(phi)
        dphi[1:-1] = 0.5*(phi[2:] - phi[:-2])
        dphi[0] = -1.5*phi[0] + 2*phi[1] - 0.5*phi[2]
        dphi[-1] = +1.5*phi[-1] - 2*phi[-2] + 0.5*phi[-3]
    else:
        dphi = np.empty_like(phi)
        dphi[:] = phi[1]-phi[0]
    return dphi


#: The path extension cannot be longer than this many knots: beyond it the spline
#: refuses the path, in the size guard inside ``SplinePath.__init__``.
_MAX_EXTENSION_KNOTS = 20000


def _extension_minimum(V, p0, dp0, sign: float, floor: float | None = None) -> float:
    """Distance to the first local minimum of ``V`` walking outward from ``p0`` along ``dp0``.

    See :ref:`note-path-deformation-extension-minimum` for the reasoning and the measurements
    behind this.
    """

    def value(x: float) -> float:
        # A potential handed to the path solver need not be defined arbitrarily far along the
        # tangent, so a probe that cannot be evaluated is treated as a wall rather than an
        # error: the walk stops at it and the end is left where it is.
        try:
            return float(np.squeeze(V(p0 + (sign * x) * dp0)))
        except errors.Timeout:
            # The run's own deadline is raised from a POSIX alarm handler, so it can land
            # inside any call to `V`. Treating it as an impassable wall would turn an aborted
            # point into a finished one, with an extension chosen from wherever the alarm fell.
            raise
        except Exception:
            return np.inf

    def signed(x: float) -> float:
        return -abs(x) if sign < 0 else abs(x)

    def clears_the_floor(x: float) -> bool:
        """Is the potential at `x` still above the other end of the path?

        See :ref:`note-path-deformation-extension-minimum-clears-the-floor` for the reasoning
        and the measurements behind this.
        """
        if floor is None:
            return True
        here = value(x)
        noise = 8.0 * np.finfo(float).eps * max(abs(here), abs(floor), np.finfo(float).tiny)
        return here > floor + noise

    # Geometric, so that a minimum arbitrarily close to the end is resolved, and reaching as far
    # as the size guard in `SplinePath` allows, so one many knots out is still found the way the
    # old unbounded search could find it. Evaluated in one call rather than point by point: the
    # potential here is the real one, the spline does not exist yet, and a few hundred scalar
    # calls to it per path end is the kind of cost that the spline exists to avoid.
    ladder = np.concatenate([[0.0], np.geomspace(1e-10, float(_MAX_EXTENSION_KNOTS), 300)])
    try:
        probes = np.asanyarray(
            V(np.asanyarray(p0)[np.newaxis, :] + (sign * ladder)[:, np.newaxis]
              * np.asanyarray(dp0)[np.newaxis, :]), dtype=float).reshape(-1)
        if probes.size != ladder.size:
            raise ValueError("the potential did not broadcast over the ladder")
    except errors.Timeout:
        # As above, and here the cost of absorbing it is 301 further scalar evaluations of a
        # potential the run has already been told to stop evaluating.
        raise
    except Exception:
        # The search below stops at the first non-finite probe, so the fallback stops filling
        # there too rather than evaluating a restricted-domain potential 300 more times.
        scalar = []
        for x in ladder:
            scalar.append(value(float(x)))
            if not np.isfinite(scalar[-1]):
                break
        probes = np.array(scalar)

    here = float(probes[0])
    if not np.isfinite(here):
        return 0.0
    best_x, best_v, previous_x = 0.0, here, 0.0
    for x, current in zip(ladder[1:], probes[1:]):
        current = float(current)
        if not np.isfinite(current):
            break
        noise = 8.0 * np.finfo(float).eps * max(abs(current), abs(best_v), np.finfo(float).tiny)
        if current > best_v + noise:
            if best_x == 0.0:
                return 0.0                      # the end is already a minimum
            found = optimize.minimize_scalar(
                value, bounds=(previous_x, float(x)), method="bounded",
                options={"xatol": 1e-10},
            )
            at = float(found.x) if found.success else best_x
            if not clears_the_floor(at):
                return 0.0      # extending there would leave no metastable end to tunnel from
            return signed(at)
        if current < best_v:
            previous_x, best_x, best_v = best_x, float(x), current
    # The walk ended without passing a minimum: it ran into a wall, hit the floor, or is still
    # falling at the far end of the ladder. In none of those cases is there a minimum to extend
    # to within reach, so the end is left where it is, which is what the old search did too on
    # a potential that keeps falling.
    return 0.0


#: Closest approach to a path end, as a fraction of the path length, of the extra samples used
#: when the barrier is too narrow for the uniform grid. It bounds how narrow a barrier the
#: potential spline can resolve.
_V_SAMPLE_END_FLOOR = 1e-10

#: How many samples must fall inside the barrier for the uniform grid to be kept. Below this
#: the grid is effectively blind to the barrier and the samples are re-placed; see
#: `_barrier_is_resolved`.
_V_SAMPLE_MIN_IN_BARRIER = 2

#: Fewest samples for which re-placing them is worth doing. The re-placed grid spans spacing
#: ratios of about 1e-9, and an interpolating cubic spline needs enough knots to carry that:
#: with `fullTunneling`'s own default of 100 samples the action came back as 1.3e+11.
_V_SAMPLE_MIN_FOR_CLUSTERING = 1000

#: How far past each path end the re-placed samples reach, counted in samples at the local
#: spacing. The spline needs a margin because the solver probes a little outside the minima.
_V_SAMPLE_OVERHANG_STEPS = 20


def _uniform_V_samples(L: float, n_samples: int) -> np.ndarray:
    """Evenly spaced samples along the path, with the 20% overhang past each end.

    This is what the potential spline has always been built on, and it is kept unchanged for
    every path whose barrier it resolves.
    """
    x = np.linspace(0, L, n_samples)
    # extend 20% beyond this so that we more accurately model the path end points
    x_ext = np.arange(x[1], L * .2, x[1])
    return np.concatenate([-x_ext[::-1], x, L + x_ext])


def _barrier_is_resolved(x: np.ndarray, y: np.ndarray, L: float) -> bool:
    """Do enough of the sampled values fall inside the barrier to represent it?

    See :ref:`note-path-deformation-barrier-is-resolved` for the reasoning and the measurements
    behind this.
    """
    inside = (x >= 0.0) & (x <= L)
    n_inside = int(np.count_nonzero(inside))
    if n_inside < 3:
        return True
    if n_inside < _V_SAMPLE_MIN_FOR_CLUSTERING:
        return True         # too few knots for the re-placed grid to be worth building
    ys = np.asarray(y, dtype=float)[inside]
    if not np.all(np.isfinite(ys)):
        return True                     # a judgement cannot be made, so nothing is changed
    level = max(float(ys[0]), float(ys[-1]))
    return int(np.count_nonzero(ys[1:-1] > level)) >= _V_SAMPLE_MIN_IN_BARRIER


def _clustered_V_samples(L: float, n_samples: int) -> np.ndarray:
    """Samples for a barrier too narrow for the uniform grid to carry.

    See :ref:`note-path-deformation-clustered-v-samples` for the reasoning and the measurements
    behind this.
    """
    n_uniform = max(2, n_samples // 2)
    # The two end grids meet at L/2, so `np.unique` drops one of that pair and the grid comes
    # out one knot short of `n_samples`. That one knot decides whether the grid is judged at
    # all: `_barrier_is_resolved` returns True without looking at anything below
    # `_V_SAMPLE_MIN_FOR_CLUSTERING` in-path knots, so at the configured 1000 samples a
    # 999-knot replacement was accepted whatever it carried. Ask for the knot back.
    n_end = max(2, (n_samples - n_uniform + 1) // 2 + 1)
    uniform = np.linspace(0.0, L, n_uniform)
    # the ends themselves are in `uniform`, so these only fill in the approach to them
    approach = np.geomspace(_V_SAMPLE_END_FLOOR * L, 0.5 * L, n_end)
    x = np.unique(np.concatenate([uniform, approach, L - approach]))

    steps = np.arange(1, _V_SAMPLE_OVERHANG_STEPS + 1)
    before = x[0] - (x[1] - x[0]) * steps[::-1]
    after = x[-1] + (x[-1] - x[-2]) * steps
    return np.concatenate([before, x, after])


class SplinePath(object):
    """Fit a spline to a path in field space, and find the potential on that path.

    See :ref:`note-path-deformation-spline-path` for the reasoning and the measurements behind
    this.

    Parameters
    ----------
    pts : array_like
        The points that describe the path, with shape ``(num_points, N_dim)``.
    V : callable
        The potential function. Input arrays will be shape ``(npts, N_dim)`` and
        output should have shape ``(npts,)``. Can be None.
    dV : callable, optional.
        The gradient of the potential. Input arrays will be shape
        ``(npts, N_dim)`` and output should have shape ``(npts, N_dim)``. Only
        used if ``V_spline_samples=None``.
    V_spline_samples : int or None, optional
        Number of samples to take along the path to create the spline
        interpolation functions. If None, the potential is evaluated directly
        from `V` given in the input. If not None, `V_spline_samples` should be
        large enough to resolve the smallest features in the potential. For
        example, the potential may have a very narrow potential barrier over
        which multiple samples should be taken.
    extend_to_minima : bool, optional
        If True, the input path is extended at each end until it hits local
        minima.
    reeval_distances : bool, optional
        If True, get more accurate distances to each knot by integrating along
        the spline.

    Attributes
    ----------
    L : float
        The total length of the path.
    """
    def __init__(self, pts: np.ndarray, V: Callable[[np.ndarray], np.ndarray],
                dV: Callable[[np.ndarray], np.ndarray]=None, V_spline_samples: int=100,
                 extend_to_minima: bool=False, reeval_distances: bool=True,
                 verbose: bool=False):
        """Construct a spline representation of the tunnelling path."""
        assert len(pts) > 1
        # 1. Find derivs
        dpts = _pathDeriv(pts)
        # 2. Extend the path
        if extend_to_minima:
            # extend at the front of the path
            # The front is the absolute minimum, which extending can only lower, so it
            # needs no floor: metastability is not at stake there.
            xmin = _extension_minimum(V, pts[0], dpts[0], -1.0)
            if xmin > 0.0: xmin = 0.0
            nx = np.ceil(abs(xmin)-.5) + 1
            if nx > _MAX_EXTENSION_KNOTS:
                raise errors.SplineError(
                    "Maximum allowed size exceeded while extending the tunnelling path towards the minimum."
                )
            if nx > 100_000 and verbose:
                msg = "WARNING: The tunneling path appears to be very long: Tried to extend it by %i points" % nx
                msg += ", which is more than 100_000. This may lead to numerical issues."
                msg += "\nThe point where this happens is at %s, with derivative = %s." % (pts[0], dpts[0])
                print(msg)
            x = np.linspace(xmin, 0, int(nx))[:, np.newaxis]
            pt_ext = pts[0] + x*dpts[0]
            
            # Sometimes there are unnecessary dimensions in the input. The
            # combination of atleast_2d and squeeze removes those and avoids
            # issues with the concatenation.
            pt_ext = np.atleast_2d(np.squeeze(pt_ext))
            pts_rest = np.atleast_2d(np.squeeze(pts[1:]))
            pts = np.concatenate([pt_ext, pts_rest], axis=0)
            
            # extend at the end of the path
            xmin = _extension_minimum(V, pts[-1], dpts[-1], +1.0,
                                      floor=float(np.squeeze(V(pts[0]))))
            if xmin < 0.0: xmin = 0.0
            nx = np.ceil(abs(xmin)-.5) + 1
            if nx > _MAX_EXTENSION_KNOTS:
                raise errors.SplineError(
                    "Maximum allowed size exceeded while extending the tunnelling path towards the second minimum."
                )
            if nx > 100_000 and verbose:
                msg = "WARNING: The tunneling path appears to be very long: Tried to extend it by %i points" % nx
                msg += ", which is more than 100_000. This may lead to numerical issues."
                msg += "\nThe point where this happens is at %s, with derivative = %s." % (pts[-1], dpts[-1])
                print(msg)
            x = np.linspace(xmin, 0, int(nx))[::-1, np.newaxis]
            pt_ext = pts[-1] + x*dpts[-1]
            pts = np.append(pts[:-1], pt_ext, axis=0)
            # Recalculate the derivative
            dpts = _pathDeriv(pts)
        # 3. Find knot positions and fit the spline.
        pdist = integrate.cumulative_trapezoid(np.sqrt(np.sum(dpts*dpts, axis=1)),
                                   initial=0.0)
        self.L = pdist[-1]
        k = min(len(pts)-1, 3)  # degree of the spline
        self._path_tck = interpolate.splprep(pts.T, u=pdist, s=0, k=k)[0]
        # 4. Re-evaluate the distance to each point.
        if reeval_distances:
            def dpdx(_, x):
                dp = np.array(interpolate.splev(x, self._path_tck, der=1))
                return np.sqrt(np.sum(dp*dp))
            pdist = integrate.odeint(dpdx, 0., pdist,
                                     rtol=0, atol=pdist[-1]*1e-8)[:,0]
            self.L = pdist[-1]
            self._path_tck = interpolate.splprep(pts.T, u=pdist, s=0, k=k)[0]
        # Now make the potential spline.
        self._V = V
        self._dV = dV
        self._V_tck = None
        if V_spline_samples is not None:
            x = _uniform_V_samples(self.L, V_spline_samples)
            y = self.V(x)
            if not _barrier_is_resolved(x, y, self.L):
                # The barrier is narrower than this grid can carry, so place the samples where
                # it is. One more evaluation of the potential, and only for such a path.
                x_clustered = _clustered_V_samples(self.L, V_spline_samples)
                y_clustered = self.V(x_clustered)
                # The re-placed grid approaches the ends geometrically, which is where the
                # barrier goes as the temperature falls, and halves the uniform samples to pay
                # for it. Around the middle of the path it is therefore the coarser of the two,
                # so a narrow barrier sitting there would be missed by the replacement as well
                # as by the original. The values are already in hand, so the same count decides
                # whether to keep it; a grid that does not resolve the barrier either is not
                # worth the spacing ratios it costs, and the uniform grid is kept instead.
                if _barrier_is_resolved(x_clustered, y_clustered, self.L):
                    x, y = x_clustered, y_clustered
                    if verbose:
                        print("The potential samples did not resolve the barrier; re-placed "
                              "them towards the ends of the path.")
                elif verbose:
                    print("Neither the uniform nor the re-placed potential samples resolve "
                          "the barrier; keeping the uniform ones.")
            self._V_tck = interpolate.splrep(x, y, s=0)

    def V(self, x):
        """The potential as a function of the distance `x` along the path."""
        if self._V_tck is not None:
            return interpolate.splev(x, self._V_tck, der=0)
        else:
            pts = interpolate.splev(x, self._path_tck)
            return self._V(np.array(pts).T)

    def dV(self, x):
        """`dV/dx` as a function of the distance `x` along the path."""
        if self._V_tck is not None:
            return interpolate.splev(x, self._V_tck, der=1)
        else:
            pts = interpolate.splev(x, self._path_tck)
            dpdx = interpolate.splev(x, self._path_tck, der=1)
            dV = self._dV(np.array(pts).T)
            return np.sum(dV.T*dpdx, axis=0)

    def d2V(self, x):
        """`d^2V/dx^2` as a function of the distance `x` along the path."""
        if self._V_tck is not None:
            return interpolate.splev(x, self._V_tck, der=2)
        else:
            raise RuntimeError("No spline specified. Cannot calculate d2V.")

    def pts(self, x):
        """
        Returns the path points as a function of the distance `x` along the
        path. Return value is an array with shape ``(len(x), N_dim)``.
        """
        pts = interpolate.splev(x, self._path_tck)
        return np.array(pts).T


@dataclass
class TunnelingSettings:
    """Configuration bundle passed around the full tunnelling workflow."""

    V_spline_samples: int | None
    tunneling_class: type
    tunneling_init_params: dict
    tunneling_findProfile_params: dict
    deformation_class: type
    deformation_init_params: dict
    deformation_deform_params: dict
    save_all_steps: bool
    verbose: bool
    callback: Callable | None
    callback_data: object


@dataclass
class TunnelingOutputs:
    """Intermediate data collected during a single deformation iteration."""

    path: SplinePath
    tobj: object
    profile1D: object
    phi: np.ndarray
    dphi: np.ndarray
    pts: np.ndarray
    deform_obj: Deformation_Spline
    converged: bool


def _build_tunneling_settings(
    V_spline_samples,
    tunneling_class,
    tunneling_init_params,
    tunneling_findProfile_params,
    deformation_class,
    deformation_init_params,
    deformation_deform_params,
    save_all_steps: bool,
    verbose: bool,
    callback,
    callback_data,
) -> TunnelingSettings:
    """Collect parameters and defaults used throughout the tunnelling routine."""
    return TunnelingSettings(
        V_spline_samples=V_spline_samples,
        tunneling_class=tunneling_class,
        tunneling_init_params=dict(tunneling_init_params or {}),
        tunneling_findProfile_params=dict(tunneling_findProfile_params or {}),
        deformation_class=deformation_class,
        deformation_init_params=dict(deformation_init_params or {}),
        deformation_deform_params=dict(deformation_deform_params or {}),
        save_all_steps=save_all_steps,
        verbose=verbose,
        callback=callback,
        callback_data=callback_data,
    )


def _fit_spline_path(pts, settings: TunnelingSettings, V, dV, verbose):
    """Fit a spline to the current path guess; translate common errors."""
    try:
        return SplinePath(
            pts,
            V,
            dV,
            V_spline_samples=settings.V_spline_samples,
            extend_to_minima=True,
            verbose=verbose,
        )
    except ValueError as err:
        if np.isnan(pts).any():
            if verbose:
                print("NaN in the path points. Returning InfiniteActionError.")
            raise errors.InfiniteActionError from err
        if verbose:
            print(err.args[0])
        raise errors.SplineError(err.args[0])
    except errors.Timeout:
        # The probes inside the path extension re-raise the run's deadline rather than reading
        # it as a wall, and it has to survive this handler too: turned into a `SplineError` it
        # reaches `bounceAction`, which treats that as no tunnelling and returns an infinite
        # action, so a timed-out point would come back as a finished one with a verdict.
        raise
    except Exception as err:
        msg = err.args[0]
        if msg == "Maximum allowed size exceeded":
            msg = (
                "Failed to fit a spline to the tunneling path. Tunneling "
                "computation failed because the number of points to "
                "construct a spline along the tunneling path is too large. "
                "This error usually happens when the points between the "
                "tunneling should appear don't match the points in the "
                "potential. Make sure that phase.valAt(T) is only called "
                "in the allowed range of temperatures."
            )
        if msg == "Invalid inputs.":
            msg = (
                "Invalid inputs to SplinePath. It might be that these are "
                "close to each other because phitol in the tunneling "
                "parameters was not set small enough: pts = %s"
            ) % (pts,)
            if np.isclose(pts[0], pts[-1]).all():
                msg += (
                    " The first and last points are identical. Return a "
                    "PotentialError with error_type 'no barrier'."
                )
                if verbose:
                    print(msg)
                raise errors.PotentialError(msg, "no barrier") from err
        if verbose:
            print(msg)
        raise errors.SplineError(msg) from err


def _create_tunneling_object(path: SplinePath, settings: TunnelingSettings):
    """Instantiate the 1D tunnelling solver for the current path."""
    if settings.V_spline_samples is not None:
        return settings.tunneling_class(
            0.0,
            path.L,
            path.V,
            path.dV,
            path.d2V,
            **settings.tunneling_init_params,
        )
    return settings.tunneling_class(
        0.0,
        path.L,
        path.V,
        path.dV,
        None,
        **settings.tunneling_init_params,
    )


def _find_profile(tobj, settings: TunnelingSettings):
    """Run the 1D tunnelling solver and normalise its exceptions."""
    try:
        return tobj.findProfile(**settings.tunneling_findProfile_params)
    except errors.TooSmalldVError as err:
        raise errors.InfiniteActionError from err
    except errors.TunnelingError as err:
        if settings.verbose:
            print(err.args[0])
        raise errors.InfiniteActionError from err


def _prepare_profile(tobj, profile1D):
    """Re-sample the instanton profile on an evenly spaced grid."""
    phi, dphi = profile1D.Phi, profile1D.dPhi
    phi, dphi = tobj.evenlySpacedPhi(phi, dphi, npoints=len(phi), fixAbs=False)
    dphi[0] = dphi[-1] = 0.0
    return phi, dphi


def _deform_path(
    path: SplinePath,
    tobj,
    profile1D,
    phi,
    dphi,
    dV,
    pts,
    settings: TunnelingSettings,
    saved_steps,
) -> TunnelingOutputs:
    """Apply the transverse deformation step and track convergence state."""
    deform_obj = settings.deformation_class(pts, dphi, dV, **settings.deformation_init_params)
    if settings.callback and not settings.callback(path, tobj, profile1D, settings.callback_data):
        if not hasattr(deform_obj, "num_steps"):
            deform_obj.num_steps = 0
        return TunnelingOutputs(path, tobj, profile1D, phi, dphi, pts, deform_obj, True)
    try:
        converged = deform_obj.deformPath(**settings.deformation_deform_params)
    except errors.DeformationError as err:
        if settings.verbose:
            print(err.args[0])
        converged = False
    pts = deform_obj.phi
    if settings.save_all_steps:
        saved_steps.append(deform_obj.phi_list)
    return TunnelingOutputs(path, tobj, profile1D, phi, dphi, pts, deform_obj, converged)


# A path that cannot reach its force target is under-resolved rather than a second branch, and
# enlarging the basis moves its action onto the curve its neighbours occupy. The improvement is
# not monotonic, though: past some size the deformation is destabilised and both the force ratio
# and the action get worse, so a loop that kept doubling would return a worse path than one that
# stopped. The loop therefore enlarges while enlarging helps and no further, and the cap holds
# the default basis at or under nb = 40.
#
# One reference, since the cap is a tuned number. Measured on commit 065ef81 with
# `models/TL_2HDM.py` class `R2HDM` at shipped settings, with the benchmark parameters of `examples/bp_lisa.yaml` at `lambda3 = 5.8205024605`, T = 15.202392 GeV, where the force ratio
# floors above its target of 0.005: nb = 10, 20 and 40 give force ratios 0.00855, 0.00799 and
# 0.00581 with `S_3/T` of 286.43, 299.92 and 299.62, while nb = 80 gives 0.09998 and 300.78, a
# force ratio an order of magnitude worse, and at T = 16.33 GeV the action collapsed by 34.
#
# The returned path is still the last iteration's, exactly as before this change; nothing here
# selects among the iterations.
_PLATEAU_IMPROVEMENT = 0.9      # less than 10% better than the previous iteration is a plateau
_MAX_BASIS_DOUBLINGS = 2        # nb 10 -> 40; 80 was measured to destabilise the deformation
_DEFAULT_BASIS_SIZE = 10        # Deformation_Spline's own default when nb is unset


def _should_stop(iter_output: TunnelingOutputs) -> bool:
    """Check if the deformation converged after a single, small adjustment."""
    return iter_output.converged and iter_output.deform_obj.num_steps < 2


def _compute_force_ratio(pts, dphi, dV, settings: TunnelingSettings) -> float:
    """Compute the ratio of transverse forces to the gradient along the path."""
    deform_obj = settings.deformation_class(pts, dphi, dV, **settings.deformation_init_params)
    F, dV_vec = deform_obj.forces()
    F_max = np.max(np.sqrt(np.sum(F * F, -1)))
    dV_max = np.max(np.sqrt(np.sum(dV_vec * dV_vec, -1)))
    return F_max / dV_max


def fullTunneling(
    path_pts,
    V,
    dV,
    maxiter=20,
    save_all_steps=False,
    verbose=False,
    callback=None,
    callback_data=None,
    V_spline_samples=100,
    tunneling_class=tunneling1D.SingleFieldInstanton,
    tunneling_init_params=None,
    tunneling_findProfile_params=None,
    deformation_class=Deformation_Spline,
    deformation_init_params=None,
    deformation_deform_params=None,
):
    """
    Calculate the instanton solution in multiple field dimensions.

    This function works by looping four steps:

      1. Fit a spline to the path given by `path_pts`.
      2. Calculate the one-dimensional tunneling along this path.
      3. Deform the path to satisfy the transverse equations of motion.
      4. Check for convergence, and then go back to step 1.

    Parameters
    ----------
    path_pts : array_like
        An array of points that constitute the initial guess for the tunneling
        path, with shape ``(num_points, N_dim)``. The first point should be at
        (or near) the lower minimum (the minimum to which the field is
        tunneling), and the last point should be at the metastable minimum.
    V, dV : callable
        The potential function and its gradient. Both should accept input of
        shape ``(num_points, N_dim)`` or ``(N_dim,)``.
    maxiter : int, optional
        Maximum number of allowed deformation / tunneling iterations.
    save_all_steps : bool, optional
        If True, additionally output every single deformation sub-step.
    verbose : bool, optional
        If True, print a message at the start of each step.
    callback : callable, optional
        User supplied function that is evaluated just prior to deforming the
        path. Should return True if the path should be deformed, and False if
        the deformation should be aborted. Should accept 4 arguments: a
        :class:`SplinePath` instance which describes the tunneling path, a
        tunneling object (instance of ``tunneling_class``), the profile found
        by the tunneling object, and extra callback data.
    callback_data : any type, optional
        Extra data to pass to the callback function if there is one.

    Other Parameters
    ----------------
    V_spline_samples : int, optional
        Passed to :class:`SplinePath`. If None, no second derivative will be
        passed to the tunneling class, and it will instead be evaluated using
        finite differences.
    tunneling_class : class, optional
        Either :class:`tunneling1D.SingleFieldInstanton` or a subclass.
    tunneling_init_params : dict, optional
        Extra parameters to pass to the tunneling class.
    tunneling_findProfile_params : dict, optional
        Extra parameters to pass to ``tunneling_class.findProfile()``.
    deformation_class : class, optional
        Either :class:`Deformation_Spline` or :class:`Deformation_Points`, or
        some other object that exposes the same interface.
    deformation_init_params : dict, optional
        Extra parameters to pass to the deformation class.
    deformation_deform_params : dict, optional
        Extra parameters to pass to ``deformation_class.deformPath()``.

    Returns
    -------
    profile1D : namedtuple
        The return value from ``tunneling_class.findProfile()``.
    Phi : array_like
        The points that constitute the final deformed path. They are in
        one-to-one correspondence with the points in `profile1D`.
    action : float
        The Euclidean action of the instanton.
    fRatio : float
        A ratio of the largest transverse force on the final path relative to
        the largest potential gradient. This would be zero if the solution were
        perfect.
    saved_steps : list
        A list of lists, with each sub-list containing the saved steps for each
        deformation. Only written to if `save_all_steps` is True.
    """
    settings = _build_tunneling_settings(
        V_spline_samples,
        tunneling_class,
        tunneling_init_params,
        tunneling_findProfile_params,
        deformation_class,
        deformation_init_params,
        deformation_deform_params,
        save_all_steps,
        verbose,
        callback,
        callback_data,
    )
    settings.deformation_init_params.setdefault("save_all_steps", save_all_steps)
    if settings.save_all_steps:
        settings.deformation_init_params["save_all_steps"] = True

    pts = np.asanyarray(path_pts)
    saved_steps = []
    last_output: TunnelingOutputs | None = None
    convergence_target = float(
        settings.deformation_deform_params.get("fRatioConv", 0.02)
    )
    previous_force_ratio = np.inf
    basis_doublings = 0

    for _ in range(1, maxiter + 1):
        # 1. Fit the spline to the path.
        path = _fit_spline_path(pts, settings, V, dV, verbose)
        # 2. Instantiate the 1D tunnelling solver and find the profile.
        tobj = _create_tunneling_object(path, settings)
        profile1D = _find_profile(tobj, settings)
        phi, dphi = _prepare_profile(tobj, profile1D)
        # 3. Deform the path to satisfy the transverse equations of motion.
        path_points = path.pts(phi)
        iter_output = _deform_path(
            path,
            tobj,
            profile1D,
            phi,
            dphi,
            dV,
            path_points,
            settings,
            saved_steps,
        )
        pts = iter_output.pts
        last_output = iter_output
        # 4. Check whether the deformation has converged.
        if _should_stop(iter_output):
            break
        # 5. If the force ratio has stopped improving while still above target, the basis
        #    cannot express the remaining correction. Enlarge it instead of taking more
        #    steps that cannot help. Measuring the force ratio costs one more deformation
        #    object and one more force evaluation per iteration, so it is measured only
        #    while a doubling is still allowed: past the cap the value could no longer
        #    change any decision, `previous_force_ratio` is never read again, and every
        #    bounce solve of every temperature would pay for it.
        if basis_doublings < _MAX_BASIS_DOUBLINGS:
            force_ratio = _compute_force_ratio(iter_output.pts, iter_output.dphi, dV, settings)
            if (
                np.isfinite(force_ratio)
                and force_ratio > convergence_target
                and force_ratio > _PLATEAU_IMPROVEMENT * previous_force_ratio
            ):
                basis_size = int(settings.deformation_init_params.get("nb", _DEFAULT_BASIS_SIZE))
                settings.deformation_init_params["nb"] = 2 * basis_size
                basis_doublings += 1
                previous_force_ratio = np.inf      # the new basis starts its own descent
                if settings.verbose:
                    print(f"Path deformation plateaued at fRatio = {force_ratio:.5g} above the "
                          f"target {convergence_target:.5g}; enlarging the spline basis to "
                          f"nb = {2 * basis_size}.")
            else:
                previous_force_ratio = force_ratio
    else:
        if settings.verbose:
            print("Reached maxiter in fullTunneling. No convergence.")

    if last_output is None:
        raise errors.InfiniteActionError

    f_ratio = _compute_force_ratio(last_output.pts, last_output.dphi, dV, settings)
    rtuple = namedtuple(
        "fullTunneling_rval", "profile1D Phi action fRatio saved_steps"
    )
    Phi = last_output.path.pts(last_output.profile1D.Phi)
    action = last_output.tobj.findAction(last_output.profile1D)
    # The shooting brackets the bounce between an undershoot and an overshoot, and at
    # convergence both must give the same action. `findProfile` records how far apart they
    # were for the profile being kept here; a large spread means the bisection stalled and
    # the profile returned is whichever trial came last. The spread grows as the transition
    # cools and the barrier becomes harder to resolve, and at the cold end one branch can
    # return a negative action while the other is positive. Reporting an unresolved bounce
    # lets the controller drop the sample, as it already does for an action it cannot
    # compute, instead of carrying a number decided by which trial ran last.
    spread = float(getattr(last_output.tobj, "bracket_action_spread", 0.0) or 0.0)
    # Written as a refused comparison rather than `spread > tol`, so that a spread which is
    # not a number, from a branch action that was not finite, is refused instead of passing.
    # The spread says the two branches agree; it does not say the number they agree on is an
    # action. A converged solve never fills the spread at all, `findProfile` records 0.0 for
    # it, so without this the whole gate is skipped for exactly the case it is least able to
    # catch. The Euclidean action of a bounce is positive. A non-positive one reaching the
    # caller is replaced there by zero and labelled `trantype` 2, a second-order transition,
    # so a solver failure would be delivered as a physical result; refused here instead, the
    # sample is dropped like any other the shooting could not resolve.
    # `np.squeeze` first, as the branch path already does before its own `float`: measured,
    # `findAction` returns a 0-d array here and `float` on that is exact, but a size-one
    # array of any other shape would raise in a future numpy and nothing here pins the shape.
    action_value = float(np.squeeze(action))
    # Finite as well as positive. `+inf > 0.0` is true, so a positive infinity passed the
    # positivity test on its own and was returned as a resolved bounce: the caller then reports
    # `trantype` 1, a first-order transition whose action is infinite, where the code's own way
    # of saying the action is infinite is `trantype` 0 with no instanton. An overflowing Simpson
    # integration on a converged profile reaches here exactly that way. The branch guard in
    # `_bracketSpread` already refuses a non-finite branch, and this makes the two agree.
    if not (np.isfinite(action_value) and action_value > 0.0):
        raise errors.InfiniteActionError(
            "The bounce is not resolved at this temperature: the action came out as "
            f"{action_value:.6g}, and the Euclidean action of a bounce is finite and positive.")
    if not (spread <= tunneling1D._ACTION_BRACKET_TOL):
        branches = getattr(last_output.tobj, "bracket_branch_actions", {}) or {}
        # `InfiniteActionError`, not `TunnelingError`: the first is the signal the callers
        # already read as "no action at this temperature" and drop the sample for, while
        # the second leaves `fullTunneling` and ends the whole point as CLI code 11. One
        # temperature the shooting cannot resolve is not a reason to abandon a coupling.
        raise errors.InfiniteActionError(
            "The bounce is not resolved at this temperature: the shooting's undershoot "
            f"gives an action of {branches.get('undershoot', float('nan')):.6g} and its "
            f"overshoot {branches.get('overshoot', float('nan')):.6g}, a relative spread "
            f"of {spread:.3g} against a tolerance of "
            f"{tunneling1D._ACTION_BRACKET_TOL:g}.")
    return rtuple(last_output.profile1D, Phi, action, f_ratio, saved_steps)


def _validate_tunneling_path(x0: np.ndarray, x1: np.ndarray, V_) -> None:
    """Sanity-check that the path spans distinct minima and tunnelling is possible."""
    if np.isclose(x1, x0, rtol=1e-5, atol=1e-8).all():
        raise errors.PotentialError(
            "It was tried to calculate tunneling between two identical minima.",
            "no barrier",
        )
    if V_(x0) <= V_(x1):
        raise errors.PotentialError(
            "V(phi_metaMin) <= V(phi_absMin); tunneling cannot occur.",
            "stable, not metastable",
        )


def _execute_full_tunneling(
    T,
    V_,
    dV_,
    tdict,
    settings,
    verbose,
    conversionFactor,
):
    """Try to compute the instanton and translate known failure modes."""
    # The straight line between the two minima. Each temperature is solved independently;
    # seeding a retry from a neighbouring temperature's converged path was measured to return
    # a non-finite action and is not part of the solver.
    initial_path = [tdict["low_vev"], tdict["high_vev"]]
    try:
        result = fullTunneling(initial_path, V_, dV_, callback_data=T, verbose=verbose, **settings)
        return {
            'instanton': result,
            'action': result.action,
            'trantype': 1,
        }, False
    except errors.PotentialError as err:
        if err.error_type == "no barrier":
            if verbose:
                print(
                    "Tunneling at T = ",
                    T * conversionFactor,
                    "GeV failed because there is no potential barrier.",
                )
            return {'action': 0, 'trantype': 2, 'instanton': None}, False
        if err.error_type == "stable, not metastable":
            if verbose:
                print(
                    "Tunneling at T = ",
                    T * conversionFactor,
                    "GeV failed because the potential is stable and not metastable.",
                )
            return {'action': np.inf, 'trantype': 0, 'instanton': None}, False
        # `err.error_type`, not `err.args[1]`: `PotentialError.__init__` passes only the
        # message to `Exception.__init__`, so `args` has one entry and `args[1]` raised
        # `IndexError` here instead of the `UnexpectedError` the line was written to raise.
        # Unreachable today, since the only two error types in the package are the two
        # handled above, and kept as the defence it was meant to be against a third.
        if verbose:
            print("Unexpected PotentialError type: ", err.error_type)
        raise errors.UnexpectedError(
            "In _execute_full_tunneling: Unexpected PotentialError type %r: %s"
            % (err.error_type, err.args[0])
        ) from err
    except errors.ZeroActionError:
        if verbose:
            print(
                "Tunneling at T =",
                T * conversionFactor,
                "GeV failed because the potential barrer is too small. Returning zero action, which corresponds to a positive infinity nucleation criterion.",
            )
        return {'action': 0, 'trantype': 2, 'instanton': None}, False
    except errors.InfiniteActionError:
        if verbose:
            print(
                "Tunneling at T =",
                T * conversionFactor,
                "GeV failed because the potential barrier is too high. Returning infinite action, which corresponds to a negative infinity nucleation criterion.",
            )
        return {'action': np.inf, 'trantype': 0, 'instanton': None}, False
    except errors.SplineError as err:
        if verbose:
            print(
                "Tunneling at T = ",
                T * conversionFactor,
                "GeV failed because the potential spline could not be constructed ("
                + err.args[0]
                + "). Treating this as no tunnelling.",
            )
        return {'action': np.inf, 'trantype': 0, 'instanton': None}, True
    except Exception as err:
        if verbose:
            print(
                "Tunneling at T =",
                T * conversionFactor,
                "GeV failed with error: ",
                err,
            )
        raise


#: How far apart two images of the false vacuum have to be, as a fraction of the length of the
#: path the bounce traverses, before they are treated as two pairings and each given its own
#: bounce.
#:
#: The criterion is the resolution of the phase tracing that produced the false vacuum. A
#: separation does not bound the difference of two bounce actions for a general potential, so
#: this is not a tolerance on the action; it is the statement that two points closer together
#: than the tracer can place a minimum are one point. The tracer locates each node by finite
#: differences of step ``deltaX_target * 1e-2``, and `phases.py` already applies that reasoning,
#: with this same constant, to the sign of a field component in ``phases._IMAGE_SIGN_RTOL``.
#:
#: An equality test at machine precision was tried and withdrawn. A symmetric phase is a false
#: vacuum at a fixed point of the group, and the tracer leaves it a few times 1e-10 from that
#: point rather than on it, so an exact test saw two images at every temperature of such a
#: transition and the solver computed one path twice. What a merge at this threshold can hide is
#: bounded by what the solver can tell apart: the two paths then differ by less than 1e-4 of
#: their length, and the solver's action on one path recomputed repeats only to a few times
#: 1e-5.
#:
#: One reference, since the threshold is a tuned number. Measured on commit 065ef81 with
#: ``models/TL_2HDM.py` class `R2HDM` at shipped settings, with the benchmark parameters of `examples/bp_lisa.yaml`` at ``lambda3 = 5.6518``, where the symmetric phase is the false
#: vacuum at all 7 temperatures that resolve: the two images lay 3.6e-13 to 2.3e-12 of the path
#: length apart, the two actions agreed to 3.0e-05 relative, and the second bounce took 49 per
#: cent of all the time spent on bounces. The tracer's own step would put the threshold at
#: 4e-06 of the path length instead, which merges the same pairs and splits the same pairs; the
#: looser constant is used so that one number serves both this test and the sign test.
_IMAGE_SEPARATION_RTOL = 1.0e-4


def _actionOfOnePairing(entry: dict) -> float:
    """The action :func:`bounceAction` stored for one pairing, as a float.

    `nan` where the entry carries no action at all, so that a missing key cannot take part
    in a comparison as if it were a number; the comparisons below refuse a `nan`.
    `bounceAction` stores a ``numpy.float64``, and ``np.squeeze`` before ``float`` because
    nothing pins the shape it arrives in.

    Parameters
    ----------
    entry : dict
        One temperature's entry as :func:`bounceAction` leaves it.

    Returns
    -------
    float
        The action, or `nan` where the entry has none.
    """
    return float(np.squeeze(entry.get("action", np.nan)))


def _isAnOrdinaryBounce(entry: dict) -> bool:
    """Whether :func:`bounceAction` resolved a first-order bounce for one pairing.

    ``trantype`` 1 with a finite positive action is a bounce. The other two outcomes are
    not: ``trantype`` 2 with action 0 says the two points it was handed coincide, and
    ``trantype`` 0 with an infinite action says the pairing gave no action at all.

    Parameters
    ----------
    entry : dict
        One temperature's entry as :func:`bounceAction` leaves it.

    Returns
    -------
    bool
        True where the entry carries a bounce the solver resolved.
    """
    action = _actionOfOnePairing(entry)
    return bool(entry.get("trantype") == 1 and np.isfinite(action) and action > 0.0)


def _isACheaperPairing(action: float, best: float) -> bool:
    """Whether `action` is the smaller of two pairings' actions.

    See :ref:`note-path-deformation-is-acheaper-pairing` for the reasoning and the measurements
    behind this.

    Parameters
    ----------
    action : float
        The candidate pairing's action.
    best : float
        The action of the pairing kept so far.

    Returns
    -------
    bool
        True where the candidate's action is the one to keep.
    """
    if not (np.isfinite(action) and action >= 0.0):
        return False
    if not (np.isfinite(best) and best >= 0.0):
        return True
    return bool(action < best)


def imagesOfTheFalseVacuum(x_false: np.ndarray, x_true: np.ndarray,
                           invGroupElements) -> list:
    r"""Every image of the false vacuum, the one nearest the true vacuum first.

    See :ref:`note-path-deformation-images-of-the-false-vacuum` for the reasoning and the
    measurements behind this.

    Parameters
    ----------
    x_false, x_true : np.ndarray
        The two minima at this temperature, as traced.
    invGroupElements : list[np.ndarray]
        The transformations the potential is invariant under, right-multiplying a point.

    Returns
    -------
    list[np.ndarray]
        The distinct images of `x_false`, nearest to `x_true` first. Never empty: the point
        as it was traced is always one of them, and it is the first entry where no other
        image is strictly nearer.
    """
    x_false = np.asarray(x_false, dtype=float)
    x_true = np.asarray(x_true, dtype=float)
    if not (np.isfinite(x_false).all() and np.isfinite(x_true).all()):
        # Nothing to choose between, and saying so here rather than leaving it to the
        # comparisons below keeps the arithmetic off a non-finite point: `x_false @ g`
        # warns on it, and a distance that is `nan` decides the question only because every
        # comparison against a `nan` is false.
        return [x_false]
    # The length of the path the bounce traverses, which is what "the same path" has to be
    # measured against. Taking the false vacuum's own length instead would make the test
    # meaningless for a false vacuum sitting at the origin, where that length is whatever the
    # tracing left behind rather than a scale of the problem, and taking an absolute
    # separation would make it depend on the units the model works in, which differ between
    # models by orders of magnitude. Where the two minima
    # coincide the path has no length and every image is the same point, so the comparison
    # falls back on the fields' own scale.
    path = float(np.linalg.norm(x_false - x_true))
    scale = path if path > 0.0 else max(float(np.linalg.norm(x_true)), 1.0)
    candidates = [(path, x_false)]
    for g in invGroupElements:
        image = np.asarray(x_false @ g, dtype=float)
        if not np.isfinite(image).all():
            continue
        # An image equal to one already accepted is the same bounce, so it is offered to the
        # solver once. The group may name a map more than once and contains the identity for
        # most models, so an image is compared against every candidate already accepted and not
        # only against the point as it was traced. Two images closer than
        # `_IMAGE_SEPARATION_RTOL` of the path are the same point to the resolution of the
        # tracing that produced them, and the note on that constant says what the test is and
        # why an exact one was withdrawn. Every image that is a distinct point is computed and
        # its action compared, which is what the function promises.
        if any(float(np.linalg.norm(image - accepted)) <= _IMAGE_SEPARATION_RTOL * scale
               for _, accepted in candidates):
            continue
        candidates.append((float(np.linalg.norm(image - x_true)), image))
    # Python's sort is stable, so two images the true vacuum is equally far from keep the
    # order they were appended in. The point as it was traced is appended first, so it stays
    # first on a tie and the answer does not depend on the order the group is given in.
    candidates.sort(key=lambda entry: entry[0])
    return [image for _, image in candidates]


def imagesToPairOver(pot) -> list:
    """The symmetry elements the bounce should be computed over, for this potential.

    See :ref:`note-path-deformation-images-to-pair-over` for the reasoning and the measurements
    behind this.
    """
    group = list(getattr(pot, "invGroupElements", []) or [])
    if not group:
        return []
    conf = getattr(getattr(pot, "config", None), "tracingConf", None)
    if not getattr(conf, "pair_transition_over_images", False):
        return []
    return group


def bounceActionOverImages(
    T: float,
    V: Callable[[np.ndarray, float], np.ndarray],
    dV: Callable[[np.ndarray, float], np.ndarray],
    outdict: dict,
    tdict: dict,
    invGroupElements=(),
    verbose: bool = False,
    conversionFactor: float = 1.0,
    **fullTunneling_params,
) -> dict:
    """Compute the bounce for every image of the false vacuum and keep the smallest action.

    See :ref:`note-path-deformation-bounce-action-over-images` for the reasoning and the
    measurements behind this.

    Parameters
    ----------
    T : float
        Temperature at which to compute the bounce action.
    V, dV : callable
        The potential and its gradient, as :func:`bounceAction` takes them.
    outdict : dict
        A dictionary mapping temperatures to tunnelling results. A result already stored
        under `T` is returned as it stands, without computing anything.
    tdict : dict
        Dictionary describing the tunnelling phase at temperature `T`, as
        :func:`bounceAction` takes it. Must contain the keys ``'low_vev'`` and
        ``'high_vev'`` for the field values of the two minima, the second of which is the
        false vacuum whose images are enumerated. Every other key is carried through onto
        the entry that is stored, so a caller that puts ``'Tnuc'`` or the phase keys in it
        gets them back.
    invGroupElements : list[np.ndarray], optional
        The transformations the potential is invariant under, right-multiplying a point.
        An empty group leaves one candidate and one bounce, the one the caller asked for.
    verbose : bool, optional
        If True, print diagnostic messages while computing the bounce action.
    conversionFactor : float, optional
        Factor to convert the temperature from internal units to GeV when printing messages.
    fullTunneling_params : dict, optional
        Extra parameters forwarded to :func:`fullTunneling`.

    Returns
    -------
    outdict : dict
        The updated output dictionary. The entry under `T` carries the usual keys, and in
        addition ``pairings_tried`` and ``pairings_resolved``, the number of images that
        were offered to the solver and the number for which it resolved a bounce, and
        ``high_vev_as_traced``, the false vacuum as the phase tracing produced it, since
        ``high_vev`` carries the image the bounce was computed on.
    """
    if T in outdict:
        return outdict

    images = imagesOfTheFalseVacuum(tdict["high_vev"], tdict["low_vev"], invGroupElements)
    best, best_action, resolved = None, float("nan"), 0
    for image in images:
        # A fresh dictionary per candidate, and `outdict` is written once at the end, so a
        # pairing that raises cannot leave a partial entry behind for the caller to read.
        trial: dict = {}
        bounceAction(T, V, dV, trial, dict(tdict, high_vev=image),
                     verbose, conversionFactor, **fullTunneling_params)
        candidate = trial[T]
        action = _actionOfOnePairing(candidate)
        if _isAnOrdinaryBounce(candidate):
            resolved += 1
        elif verbose:
            print("Tunneling at T = ", T * conversionFactor,
                  "GeV gave no bounce on one image of the false vacuum: trantype ",
                  candidate.get("trantype"), " action ", action)
        if best is None or _isACheaperPairing(action, best_action):
            best, best_action = candidate, action
    best["pairings_tried"] = len(images)
    best["pairings_resolved"] = resolved
    # The false vacuum as the phase tracing produced it, beside the image the bounce was
    # computed on. A caller holding this entry and the two traced phases cannot otherwise tell
    # that it belongs to them: the entry's `high_vev` is the winning image, which for a group
    # that reverses the fields is a reflection of the traced point rather than the point
    # itself, and `bubbledynamics` compares the two to decide whether a cached action can be
    # reused.
    best["high_vev_as_traced"] = tdict["high_vev"]
    outdict[T] = best
    return outdict


def bounceAction(
    T: float,
    V: Callable[[np.ndarray, float], np.ndarray],
    dV: Callable[[np.ndarray, float], np.ndarray],
    outdict: dict,
    tdict: dict,
    verbose: bool = False,
    conversionFactor: float = 1.0,
    **fullTunneling_params,
) -> dict:
    """
    Compute the 1D bounce action and update the tunnelling dictionaries.

    Parameters
    ----------
    T : float
        Temperature at which to compute the bounce action.
    V : callable
        Potential function. Input arrays have shape ``(num_points, N_dim)`` and
        the output should have shape ``(num_points,)``.
    dV : callable
        Gradient of the potential with the same input shape as `V`. The output
        should have shape ``(num_points, N_dim)``.
    outdict : dict
        A dictionary mapping temperatures to tunnelling results. If the
        result for temperature `T` is already in this dictionary, it is
        returned immediately.
    tdict : dict
        Dictionary describing the tunnelling phase at temperature `T`. Must
        contain the keys ``'low_vev'`` and ``'high_vev'`` for the field values of
        the two minima.
    verbose : bool, optional
        If True, print diagnostic messages while computing the bounce action.
    conversionFactor : float, optional
        Factor to convert the temperature from internal units to GeV when
        printing messages.
    fullTunneling_params : dict, optional
        Extra parameters to pass to :func:`fullTunneling`. The function accepts
        them via ``**fullTunneling_params`` and forwards the dictionary.

    Returns
    -------
    outdict : dict
        The updated output dictionary, with the result for temperature `T`
        stored under that key.
    """
    if T in outdict:
        return outdict

    def V_(x, T=T, V=V):
        return V(x, T)

    def dV_(x, T=T, dV=dV):
        return dV(x, T)

    x1, x0 = tdict['low_vev'], tdict['high_vev']

    try:
        _validate_tunneling_path(x0, x1, V_)
    except errors.PotentialError as err:
        if verbose:
            print(
                "Tunneling at T = ",
                T * conversionFactor,
                "GeV failed during path validation: ",
                err.args[0] if err.args else err.error_type,
            )
        if err.error_type == "no barrier":
            tdict['action'] = 0
            tdict['trantype'] = 2
            tdict['instanton'] = None
        elif err.error_type == "stable, not metastable":
            tdict['action'] = np.inf
            tdict['trantype'] = 0
            tdict['instanton'] = None
        else:
            raise errors.UnexpectedError(
                "In bounceAction: the potential failed with an error type this routine does "
                "not handle, %r, saying: %s. The handled types are 'no barrier' and "
                "'stable, not metastable'." % (err.error_type, err.args[0] if err.args else "")
            ) from err
        outdict[T] = tdict
        return outdict

    # Run the full tunnelling routine and translate the outcome into
    # the structure expected by the downstream percolation code.
    result, early_exit = _execute_full_tunneling(
        T,
        V_,
        dV_,
        tdict,
        fullTunneling_params,
        verbose,
        conversionFactor,
    )

    tdict['instanton'] = result.get('instanton')
    tdict['action'] = result['action']
    tdict['trantype'] = result['trantype']

    # Unreachable since `fullTunneling` began refusing an action that is not finite and
    # positive: `_execute_full_tunneling` returns `action` as 0, `np.inf`, or the value that
    # guard let through, and the two assignments above this one are in a branch that returns
    # before reaching here. Kept as defence against a future caller, not as a live path.
    if tdict['action'] < 0:
        if verbose:
            print(
                "Tunneling at T = ",
                T * conversionFactor,
                "GeV failed because the action was computed as negative. This can happen if the potential barrier is extremely shallow. Returning zero action, which corresponds to a second-order transition.",
            )
        tdict['action'] = 0
        tdict['trantype'] = 2

    outdict[T] = tdict

    # When a spline failure occurs we skip downstream calculations, but still
    # provide the partial dictionary to avoid repeated work.
    if early_exit:
        return outdict

    return outdict


def bounceSolution(pot, T: float, xstart: np.ndarray, xend: np.ndarray,
                   verbose: bool = False):
    """Compute the instanton solution for tunneling from `xstart`
    to `xend`. Return the action and the instanton solution
    (bubble profile).

    Parameters
    ----------
    pot : generic_potential
        Effective potential 
    T : float
        Temperature at which to compute the bounce action.
    xstart: np.ndarray
        high temperature minimum
    xend: np.ndarray
        low temperature minimum
    verbose : bool, optional
        If True, print diagnostic messages while computing the bounce action.

    Returns
    -------
    profile1D : namedtuple
        The return value from ``tunneling_class.findProfile()``.
    Phi : array_like
        The points that constitute the final deformed path. They are in
        one-to-one correspondence with the points in `profile1D`.
    action : float
        The Euclidean action of the instanton.
    fRatio : float
        A ratio of the largest transverse force on the final path relative to
        the largest potential gradient. This would be zero if the solution were
        perfect.
    saved_steps : list
        A list of lists, with each sub-list containing the saved steps for each
        deformation. Only written to if `save_all_steps` is True.
    """

    def V_(x, T=T):
        return pot.Vtot(x, T, include_radiation=False, include_decoupled=False)

    def dV_(x, T=T):
        return pot.gradV(x, T)

    result = fullTunneling([xend, xstart], V_, dV_, callback_data=T,
                           **pot.config.tracingConf.tunneling_params)

    return result
