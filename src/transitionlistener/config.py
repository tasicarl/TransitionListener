"""This module contains the configurations.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

import copy


class TracingConf:
    """Phase-tracing and tunneling computation parameters."""

    internal_scale = 1000      # Reference energy scale (internal units); sets the magnitude of the stable minimum.
    Tmax_factor = 2.5               # Tmax_factor * internal_scale = max tracing temperature.
    diftol = 1                      # Max field-space distance between two phases (internal units) before they merge.
    Z2_cutof_factor = -5            # For Z2-symmetric potentials: reject phases with field < Z2_cutof_factor * diftol.
    tracing_derivative_order = 4    # Finite-difference derivative order: 2 or 4.
    tracing_field_accuracy = 1e-3   # Target field-space accuracy during tracing (1e-3 works well).
    tracing_temp_accuracy = 1e-3    # Target temperature accuracy during tracing.
    gen_mirror_phases = False       # If True, also generate mirror-transformed phases of the potential.
    nucleation_Ttol = 1e-8          # Nucleation-temperature tolerance (internal units).
    do_tachyon_test = True          # If True, verify the potential is stable at T = 0.
    approx_strength_threshold = 1e-4  # Transitions with DV / T^4 below this are treated as second-order.
    do_high_T_phase_check = True    # Hotfix toggle for high-T phase checks (needed by the vev-flip-flop model).

    tracing_args = {
        "dtstart": 1e-4,         # Initial trace step for traceMultiMin (scaled by tHigh - tLow).
        "tjump": 1e-5,           # Branch-jump size when tracing continues after a phase endpoint.
        "dtabsMax": 20.0,        # Max absolute step is largest of abs(dtstart)*dtabsMax ...
        "dtfracMax": 0.25,       # ... and t*dtfracMax in internal units.
        "dtmin": 1e-6,           # Min step (relative to dtstart) before assuming transition ends.
        "deltaX_tol": 1.2,       # deltaX_tol*deltaX_target = max x-error before shrinking the step.
        "minratio": 1e-4,        # Smallest/largest Hessian eigenvalue ratio before branch ends.
        "local_min_args": {      # Forwarded to phases.findApproxLocalMin() from traceMultiMin().
            "n": 100,            # Interpolation samples along x1->x2 for local-min detection.
            "edge": 0.05,        # Fraction trimmed off each end before searching.
        },
    }
    """Arguments forwarded to the phase-tracing routines."""
    tunneling_params = {
        "tunneling_findProfile_params": {
            "phitol": 1e-10,     # Field-space tolerance for the overshoot/undershoot profile.
            "xtol": 1e-10,       # Radial shooting tolerance for the bounce match.
            "rmin": 1e-4,
            "rmax": 1e4,
            "npoints": 500,      # Radial support points for the bounce profile.
        },
        "V_spline_samples": 1000,  # Points used to spline V(x) along the bounce path; None evaluates V directly.
        "tunneling_init_params": {
            "phi_eps": 1e-6,     # Initial offset from the false vacuum.
            "rscale": None,      # Manual radial scale; None = auto-estimate.
        },
        "deformation_deform_params": {
            "startstep": 2e-3,
            "fRatioConv": 2e-2,  # Convergence on normal/tangential force ratio along the path. Auto-tightened to 1e-2 for Ndim>=2 models in generic_potential.__init__.
            "converge_0": 5.0,   # Early-stop threshold for the force-ratio at the first stage.
            "fRatioIncrease": 5.0,
            "maxiter": 500,
            "verbose": False,
        },
    }
    """Arguments forwarded to the bounce-action / tunneling routines."""

    def __init__(self):
        """Give every model its own copy of the mutable default dicts."""
        self.tracing_args = copy.deepcopy(TracingConf.tracing_args)
        self.tunneling_params = copy.deepcopy(TracingConf.tunneling_params)


class PercolationConf:
    """Settings for the percolation solver."""

    # Solver mode and integral backend.
    algorithm_mode = "adaptive_step_size"       # "adaptive_step_size" (default) or "fixed_step_size".
    integral_method = "ode"               # Percolation integral backend: "ode" or "double_integral".
    time_temperature_mode = "sound_speed" # dT/dt relation: "sound_speed" or "bag".

    # Percolation targets.  Iperc is computed as -ln(1 - f_perc) at use sites.
    f_perc = 0.28957                      # True-vacuum fraction at percolation.
    f_start = 1e-3                        # Max initial true-vacuum fraction; below this, widen the T-range.
    f_final = 0.99                        # Min final true-vacuum fraction for a completed percolation.

    # Iteration controls shared by both percolation passes.
    weight = 1 / 1.5                      # Fraction of support points spent between Tnuc and Tperc (fixed_step_size mode).
    maxit = 10                            # Max iterations per percolation pass (step 2 and step 3 each).
    rel_increment = 0.10                  # Relative T-range growth when the initial range is too small.
    max_boundary_ratio = 0.45             # Max share of T-range allowed on plateaus where P_true = 0 or 1.

    # Support-bank budget (adaptive step size mode).
    n_action_min = 15                     # Min support points built before a DZW sweep.
    n_action_increment = 5                # Support points added per DZW refinement iteration.
    n_action_max = 60                     # Hard cap on total support points across DZW iterations.
    max_action_temperatures = 100         # Absolute ceiling on distinct action evaluations per run.

    # fixed_step_size support count.
    n_action = 30                         # Fixed support count used by the fixed_step_size path.

    # P(T)-jump refinement and final acceptance.
    large_delta_p_refine_threshold = 0.1  # Neighbour DeltaP above this triggers extra refinement.
    large_delta_p_success_threshold = 0.2 # Neighbour DeltaP above this blocks acceptance.

    # Diagnostics and optional rescue paths.
    jitter_GH4_threshold = 1.0            # log10(Gamma/H^4) jitter tolerance before flagging unresolved action.
    jitter_rescue = True                  # If True, rerun with tunneltight precision on jitter detection.
    n_jitter_save = 20                    # Max jitter diagnostic samples saved to disk.

    # Observable accuracy targets (step 2 and step 3).
    acc_tperc = 1e-2                      # Relative accuracy goal on Tperc.
    acc_tfinal = 1e-2                     # Relative accuracy goal on Tfinal.
    acc_rh = 1e-2                         # Relative accuracy goal on RH.

    # Strength-based validity flag.
    weak_threshold = 1e-4                 # Below this alpha at Tperc, the percolation result is flagged invalid.


class GWConf:
    """Settings for calculating the gravitational-wave signal."""

    use_mean_bubble_separation = True  # If True, use mean bubble separation for the GW signal scale.
    wall_velocity = "LTE"              # "LTE", "c", "WallGo", or a float in (0, 1].
    bw_collisions = "off"              # Bubble-wall collisions: "off", "full" (kappa_phi=1), or "NLO".
    turbulence = "off"                 # Turbulence contribution: "off" or a model key.
    dilution = False                   # If True, apply a dilution factor to the GW signal.
    equilibrium_DS = True              # Assume the dark sector is in thermal equilibrium with the SM.
    epsilon_turbulence = 0.1           # Energy fraction redirected into turbulence (0 disables).
    sound_speed = "compute"            # Sound-speed strategy: "compute" (broken-phase cs) or "1/3" (bag value).
    coupled_hydrodynamics = True       # Couple the dark and visible sectors in the hydro solve.
    check_if_T0_global_min = False     # If True, require the T=0 minimum to be global (else just warn).


class Configuration:
    """Container holding one independent configuration object per subsystem."""

    def __init__(self):
        """Start every model with independent configuration objects."""
        self.tracingConf = TracingConf()
        self.gwConf = GWConf()
        self.percolationConf = PercolationConf()


all_observables = {
    "alpha": R"$\alpha$",
    "alpha_theta": R"$\alpha_{\theta}$",
    "alpha_thetabar": R"$\alpha_{\bar\theta}$",
    "alpha_inf": R"$\alpha_\infty$",
    "alpha_eq": R"$\alpha_\mathrm{eq}$",
    "betaH_S3": R"$(\beta /H)_{S_3}$",
    "betaH_RH": R"$(\beta /H)_{RH}$",
    "RH": R"$RH$",
    "Treh_SM_GeV": R"$T^{\mathrm{reh}}_{\mathrm{SM}}$ / $\mathrm{GeV}$",
    "Tperc_SM_GeV": R"$T^{\mathrm{perc}}_{\mathrm{SM}}$ / $\mathrm{GeV}$",
    "g_eff_tot_reh": R"$g^{\mathrm{reh}}_{\mathrm{tot}}$",
    "h_eff_tot_reh": R"$h^{\mathrm{reh}}_{\mathrm{tot}}$",
    "kappa_phi": R"$\kappa_{\phi}$",
    "kappa_sw": R"$\kappa_{\mathrm{sw}}$",
    "kappa_turb": R"$\kappa_{\mathrm{turb}}$",
    "g0": R"$g_0$",
    "h0": R"$h_0$",
    "v_wall": R"$v_{\mathrm{wall}}$",
    "D": R"$D$",
    "c_s": R"$c_\mathrm{s}$",
    "c_s_sym": R"$c_\mathrm{s}^\mathrm{sym}$",
    "c_s_bro": R"$c_\mathrm{s}^\mathrm{bro}$",
    "step": R"$\mathrm{step}$",
    "total_steps": R"$\mathrm{total\ steps}$",
    "Tnuc_SM_GeV": R"$T^{\mathrm{nuc}}_{\mathrm{SM}}$ / $\mathrm{GeV}$",
    "Tcrit_SM_GeV": R"$T^{\mathrm{crit}}_{\mathrm{SM}}$ / $\mathrm{GeV}$",
    "Tf_SM_GeV": R"$T^{\mathrm{final}}_{\mathrm{SM}}$ / $\mathrm{GeV}$",
    "xi_crit": R"$\xi_{\mathrm{c}}$",
    "WARNING:too_weak_to_compute_perc": R"$\mathrm{WARNING:} \mathrm{too\ weak\ to\ compute\ perc}$",
    "WARNING:no_perc_splines": R"$\mathrm{WARNING:} \mathrm{no\ perc\ splines}$",
    "WARNING:betaH_small": R"$\mathrm{WARNING:} (\beta/H)_{RH} \mathrm{\ too\ small}$",
    "WARNING:betaH_very_small": R"$\mathrm{WARNING:} (\beta/H)_{RH} \mathrm{\ very\ small}$",
    "WARNING:betaH_mismatch": R"$\mathrm{WARNING:} (\beta/H)_{S_3} \mathrm{\ vs\ } (\beta/H)_{RH} \mathrm{\ mismatch}$",
    "WARNING:betaH_nonfinite": R"$\mathrm{WARNING:} (\beta/H)_{S_3} \mathrm{\ or\ } (\beta/H)_{RH} \mathrm{\ nonfinite}$",
    "WARNING:nucleationRate_nonexponential": R"$\mathrm{WARNING:} \mathrm{nucleation\ rate\ nonexponential}$",
    "WARNING:spline_tnuc_unavailable": R"$\mathrm{WARNING:} T_\mathrm{nuc} \mathrm{\ spline\ unavailable}$",
    "WARNING:spline_tnuc_not_reached": R"$\mathrm{WARNING:} T_\mathrm{nuc} \mathrm{\ criterion\ not\ reached}$",
    "WARNING:spline_tnuc_failed": R"$\mathrm{WARNING:} T_\mathrm{nuc} \mathrm{\ spline\ failed}$",
    "WARNING:not_T0_global_min": R"$\mathrm{WARNING:} T=0 \mathrm{\ not\ global\ min}$",
}
