# Changelog

All notable changes between releases of TransitionListener.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Output column `Treh_DS_GeV`: the reheating temperature of the
  transitioning sector, i.e. the fields of the potential and the coupled
  radiation bath.
- Model attribute `SM_bath` (`"coupled"` or `"decoupled"`) that names the
  radiation bath holding the Standard Model. If it is not set, TL takes the
  bath whose tables are `e_geffSM`, and warns if both are. A model with
  Standard Model fields in its potential (e.g. the 2HDM) cannot decouple the
  Standard Model; TL raises an error when it is set up that way.
- `constants.h_eff_today`, today's entropy degrees of freedom.

### Changed

- **Sound speed in the pseudo-trace**: the pseudo-trace strengths now divide
  the pressure of *both* phases by the broken-phase sound speed, as in
  arXiv:2004.06995 eq. (2.13), arXiv:2010.09744 and arXiv:2206.01130 sec. 2.
  Each phase was divided by its own, which leaves whatever the two phases
  share, a radiation bath or the zero point of the potential, in the
  difference. Affected: `alpha` and `alpha_hyd` in both solvers,
  `alpha_thetabar` in the adaptive solver, and `alpha_theta` in the fixed step
  size one, where that column holds the pseudo-trace strength; in the adaptive
  solver `alpha_theta` is the bag-model strength and is unchanged, as are
  `alpha_p` and `alpha_e`. `Tperc`, `Treh`, `beta/H` and the peak frequency are
  unchanged, the amplitude is not. The sound speed of the percolation
  integral's time-temperature relation is a different quantity and is
  untouched.

- **Redshift of the gravitational-wave spectrum**: `Treh_SM_GeV` is the
  temperature of the Standard Model bath at reheating: the reheated
  temperature if the Standard Model is coupled to the transitioning sector,
  `Tperc` if it is decoupled; before, it always held the reheating temperature
  of the transitioning sector. `g_eff_tot_reh` and `h_eff_tot_reh` sum the
  fields of the potential, the coupled bath and the decoupled bath, each at its
  own temperature after reheating and weighted by its temperature ratio to the
  Standard Model bath, `(T_i/T_SM)^4` and `(T_i/T_SM)^3` (arXiv:2109.06208,
  arXiv:2311.06346), with the radiation parts taken from `kin_coupled_*` and
  `kin_decoupled_*`. Before, they were evaluated at `Tperc` but paired with the
  reheating temperature; with a coupled Standard Model they are now evaluated
  at the reheating temperature (arXiv:2502.19478). The energy and entropy of a
  decoupled dark sector are assumed to end up in the photon bath without
  entropy injection (`D = 1`). This changes spectra and signal-to-noise ratios
  also without a decoupled bath: by a fraction of a per cent where the degrees
  of freedom barely change between `Tperc` and `Treh`, but by up to a factor
  1.36 in peak frequency and 0.75 in amplitude on the strongly supercooled
  conformal dark U(1) line (`y = 0.01`, `v = 0.14 GeV`), where `Tperc` falls
  to 40 keV and `Treh` is 10.6 MeV.
- `g_eff_tot_reh` and `h_eff_tot_reh` count the Standard Model fields of the
  potential. The 2HDM models put W, Z, t, h and further Standard Model fields
  into the potential and flag them `is_SM`; these were masked out while the
  Standard Model tables left no room for them, so they were counted nowhere,
  and `g_eff_tot_reh` came out near 69 instead of near 98. The 2HDM spectra
  move by about +6 % in peak frequency and -10 % in amplitude; models without
  Standard Model fields in the potential are unaffected.
- Today's entropy degrees of freedom `h0` passed to the spectrum were 3.91;
  they are now 3.9309 (arXiv:1803.01038), the value `gwfopt` already used as
  its default. All amplitudes rise by 0.7 %.

### Fixed

- **A broken phase with no thermal pressure**: at percolation temperatures far
  below the mass scale, the thermal part of the potential underflows in the
  broken phase, so its enthalpy and its sound speed come back as zero or as not
  a number. Two places carried that forward. The wall velocity in local thermal
  equilibrium compared the strength and the enthalpy ratio against bounds, which
  a not-a-number passes, and then integrated the plasma from a non-finite initial
  state, raising an error from the integrator; it now returns a wall velocity of
  one, since a broken phase without a plasma gives the wall nothing to push
  against. The pseudo-trace strengths `alpha_thetabar` and `alpha_hyd` divided by
  the broken-phase sound speed and raised a division by zero; they are now not a
  number where no sound speed exists, with the reason reported in verbose mode.
  The strengths that need no sound speed, `alpha_p`, `alpha_theta`, `alpha_e`,
  `alpha_inf` and `alpha_eq`, are unaffected. The fixed step size solver has its
  own copy of that calculation, where the same division produced an infinity and
  then a not-a-number together with a floating-point warning rather than an
  error, and where the pseudo-trace strength is called `alpha_theta`; it is
  guarded in the same way. No result changes where the broken-phase plasma
  exists.

- **Degrees of freedom of the fields of the potential**: `h_eff_DS` and
  `g_eff_DS`, used for the temperature inside the bubbles, for the integration
  of `Treh` and for the redshift, masked out every field flagged `is_SM`,
  skipped the Goldstone modes and kept the ghosts of the gauge bosons. They now
  count every mode and subtract the ghosts, i.e. the Landau gauge counting that
  `radiationEnergyDensity` has always used. For the 2HDM models W, Z, t, b,
  tau, h and the Goldstone modes were counted nowhere, since the Standard Model
  tables were capped below them: 68 entropy degrees of freedom at 50 GeV
  instead of 93. Where the Goldstone modes are massless at the minimum, as in
  `models/TL_dark_U1.py`, the counting is unchanged to machine precision, since
  the Goldstone that was dropped and the ghost that was kept cancel; where they
  are not, as in `models/TL_conformal_dark_u1.py`, it changes, by -16 % at
  `T = 0.1 v`. The count is floored at zero: the ghosts are massless in the
  Landau gauge while the Goldstone modes they cancel are not, so once every mode
  of the potential is frozen out the difference would otherwise turn negative.
  `reheating_geff` now also evaluates the fields of the potential with their
  zero-temperature masses, as `radiationEnergyDensity` and `h_eff_DS` do, rather
  than with the Debye-corrected ones.
- **Standard Model fields of the potential in the tabulated baths**: the
  Standard Model tables were frozen above half the mass of the lightest massive
  `is_SM` field of the potential (`set_sm_temperature_cap`, 0.888 GeV for the
  2HDM) to leave room for those fields. The cap also froze the photon, the
  gluons, the light fermions and the whole QCD crossover above that
  temperature, and it was module-wide state that the construction of one
  potential changed for all others. It is replaced by subtracting the Standard
  Model fields of the potential from the tables, at the masses they have in the
  zero-temperature vacuum, so that each field is counted exactly once: in the
  zero-temperature vacuum the total is the full Standard Model table plus the
  additional fields of the model. `set_sm_temperature_cap` is removed.
  Together with the previous entry, this lowers the degrees of freedom after
  reheating of the 2HDM benchmark line by 2.9 % in energy and 1.8 % in entropy,
  moves `Tperc` by 0.02 % and `Treh` by 0.04 %, and `alpha`, which is built from
  the sound speed, by 1.4 %; along the dark lines the percolation is unchanged
  and only the redshift moves, by -0.87 % (conformal) and +0.29 % (abelian dark
  Higgs) in the degrees of freedom after reheating.
- **Transverse photon mass in `models/TL_2HDM.py`**: the transverse photon was
  given the Debye mass of the longitudinal photon instead of zero, since both
  eigenvalues were taken from the neutral gauge mass matrix including the Debye
  terms. The transverse modes receive no thermal mass at this order, and that
  eigenvalue vanishes identically without them. The wrong mass entered the
  thermal potential and, with two degrees of freedom, the daisy term, so it
  shifted the barrier of every 2HDM run, `TL_2HDM_BSMPT.py` included. Along the
  benchmark line `Tcrit` falls by 0.26 %, `Tperc` by 0.35 %, `Treh` by 0.27 %,
  `alpha` rises by 1.4 % (up to 7.5 %) and the peak amplitude by 3.7 %. The step
  in `alpha` and `beta/H` at `lambda3 = 5.69`, which the daisy term produced, is
  gone. One point of the 49-point line ends in error 10 where it had a result
  before, and does not come back with more support points; its neighbours are
  unaffected.
- **Scale factor of the percolation integral**: `a(T)` was obtained by
  integrating `d ln a / dT = -1 / (3 c_s^2 T)` with the trapezoidal rule over the
  solver's temperature grid, which spans the overlap of the two traced phases and
  can run over ten or more decades with few points. Each interval was weighted by
  the sound speed at its cold end, where the false vacuum is far below completion
  and its sound speed is meaningless, so single intervals contributed hundreds of
  e-folds: for the 2HDM benchmark `a(T)` jumped by 10^135 between neighbouring
  grid points and reached 10^217, and the percolation integral then failed with
  error 10. It now uses the integral form of the same relation, entropy
  conservation `a^3 s = const`, with the bag relation `a ~ 1/T` where the entropy
  of the traced phase is not usable, and it accepts sound speeds only inside
  `(0, 1]`. For a pure radiation bath, where `a ~ 1/T` exactly, the old rule was
  wrong by a factor 680 at the cold end of a twelve-decade grid. Along the 2HDM
  benchmark line the change moves `Tperc`, `Treh` and `alpha` by less than
  0.01 %, and it gives back the one point that the transverse photon fix had
  cost, so the line has a result at all 49 of its points.
- **Expansion history of the mean bubble separation**: the bubble number density
  `n_B` was integrated with the bag relation, `a ~ 1/T` and `dT/dt = -H T`, while
  the percolation integral that fixed `Tperc` used the sound speed of the plasma
  and an entropy-conserving `a(T)`, so `R_*` and `Tperc` described two different
  cosmologies. `calcMeanBubbleSeparation` already accepted the history through
  `entropyInt` and `coolingInt`, but nothing passed it. Both now come from
  `expansion_interpolants`, built from the same routine the percolation integral
  uses, so `Tperc`, `R_*` and `beta/H` belong to one expansion history. With
  `C = 1/(3 c_s^2)` at `Tperc`, the old expression gave an `R_* H_*` too large by
  `C^(1/3)`: for an exact exponential nucleation rate, `beta/H` read off `R_*`
  keeps the same residual for `C = 1`, 1.08 and 1.2 once the history is shared,
  and drifts by -1.7 % and -5.1 % without it. `Tperc`, `alpha`, `kappa_sw` and
  `(beta/H)_S3` are unchanged; `R_* H_*`, `(beta/H)_RH` and the spectrum are not.
  Along the abelian dark Higgs line (`C` from 0.974 to 1.090) `R_* H_*` moves by
  a median -2.20 % (range -2.69 % to +0.83 %) and the peak amplitude by -4.34 %
  (-5.31 % to +1.64 %); along the supercooled conformal line by -0.15 % (-2.06 %
  to +0.83 %) and -0.31 %; along the 2HDM line by +0.28 % (-0.32 % to +1.01 %)
  and +0.57 %. At single points: -2.51 % in `R_* H_*` for the abelian dark Higgs
  at `lambda = 0.03`, -4.67 % for the conformal model at `v = 20 GeV`
  (`C = 1.175`) and +0.92 % for the 2HDM benchmark, where `C = 0.972` is below
  one and `R_*` grows instead. Every point keeps its error code, and the fixed
  step size solver, which uses the bag relation throughout, is unchanged.
  The expansion history is carried as `ln s = -3 ln a` rather than as `s` itself,
  and the ratio the separation needs is formed as a difference of logarithms:
  `a^-3` underflows to zero beyond about 236 e-folds, while the scale factor is
  allowed 700, and a ratio of two underflowed entropies comes back as one, an
  unexpanding universe. The integral over the separation also moved from a linear
  to a logarithmic temperature grid, which matters where the percolation support
  spans decades: at a support ratio of 1000 the linear grid was 1.6 % low at the
  same number of points, at a ratio of 100 it was 0.10 % low. The `bag` mode keeps
  the linear grid and the bag relation throughout and is unchanged.
- **Sound speed of the `(beta/H)_S3` time-temperature factor**: the factor `3 c_s^2`
  that converts `T d(S_3/T)/dT` into `(beta/H)_S3` was read from the sound speed of the
  gravitational wave settings, `GWConf.sound_speed`. With `GWConf.sound_speed = "1/3"`
  that factor became exactly one, so `(beta/H)_S3` lost the correction while the
  percolation history kept the sound speed of the plasma. It now comes from the
  percolation history itself, through the same routine the percolation integral,
  the mean bubble separation and the false-vacuum criterion use. With the default
  `GWConf.sound_speed = "compute"` the two agree to 2e-14 and no result changes.
- **Double integral and the time-temperature mode**: with
  `percolation_integral_method = "double_integral"`, `percIntegral` ignored the
  expansion history whatever `percolation_time_temperature_mode` was set to,
  while `(beta/H)_S3` still carried its `3 c_s^2` factor. `percIntegral` now
  implements `entropy_density` and `cooling_factor` and receives the same factors
  as the ODE; it reduces exactly to the previous expression in the bag limit, and
  agrees with the ODE to better than 1e-3 wherever the percolation integral is of
  order one. For the conformal model at `g = 0.692`, `Tperc` moves by +0.46 %,
  +0.64 % and +1.41 % at `v = 2`, 6 and 20 GeV, `R_* H_*` by +5.7 %, +7.5 % and
  +15.6 %, and the peak amplitude by +11.6 %, +15.6 % and +33.7 %. The default
  method (`"ode"`) is unaffected. One rule, `percolation_uses_sound_speed`, now
  decides for the percolation integral, `R_*`, `(beta/H)_S3` and
  `percolation_sound_speed_sq` alike; the sound speed is used unless the mode is
  `"bag"`, for both integral methods.
- **Diagnostic plots**: `plotDOFs` drew the fields of the potential against the
  whole Standard Model table, so its total double counted the Standard Model
  fields of the potential; both curves now use the counting of the solvers.
  `plotPercolation` unpacked nine return values from `calcPercAndEvolve`, which
  returns six, so it failed for every model; the three extra values were never
  used. It also failed when a transition has no nucleation temperature, where
  the upper end of the percolation profile now sets the plotting range.
- **Decoupled radiation baths**: runs with radiation in `kin_decoupled_*`,
  the setting 2.0.0 prescribes for a sector decoupled from the Standard Model,
  ended with error code 8 or gave wrong results without an error. The
  percolation solvers mixed the decoupled bath into the reheating of the
  bubbles: the energy balance counted it in the released energy but not in the
  broken phase, the expansion rate evaluated it at the temperature inside the
  bubbles, and the entropy conservation between support points and the
  integration of `Treh` added the Standard Model entropy table whatever the
  baths were. Now only the transitioning sector is reheated; the decoupled bath
  keeps the temperature of the surrounding false vacuum and enters the
  expansion rate. This holds in the adaptive solver including its
  self-consistency check, in the fixed step size solver of the pipeline
  (`bubbledynamics_fixedstep`) and in the one reached through
  `bubbledynamics.calcPercAndEvolve` (`percolation_fixedstepsize`). The
  `radiationEnergyDensity` of the BSMPT 2HDM models now includes a decoupled
  bath like the base class. Percolation results without a decoupled bath are
  unchanged.
- In `h2Omega_0_sum` the break frequencies were redshifted with `D^(-4/3)`
  instead of `D^(-1/3)`; the amplitude keeps `D^(-4/3)`. No effect for the
  default `D = 1`.
- **Treh scatter (#6)**: since 2.1.0 the reheating temperature was read off the
  tabulated broken-phase temperature trajectory at `Tperc`, which made it depend
  on where the adaptive support points happen to sit and produced
  point-to-point scatter along smooth parameter scans. The broken-phase
  temperature is now integrated down to `Tperc` with an error-controlled solver
  (`integrate_broken_temperature`), in both the adaptive and the fixed step size
  percolation solver; the true-vacuum fraction enters through its
  percolation integral `I = -ln(1 - P)`, whose logarithm is smooth, instead of
  through a spline of `P`. The tabulated trajectory and the instantaneous
  reheating solve remain as fallbacks. The read-off also carried a bias that
  shrinks only linearly with the support spacing, so `Treh` moves down by a few
  hundredths to a few tenths of a per cent, depending on the model.
- **Tperc** in the adaptive step size solver is now taken from an event of the
  percolation-integral ODE rather than from root-finding on interpolated
  samples, so it no longer moves with the support points.
- A degenerate step-3 profile, which reaches percolation without any evolved
  broken-phase temperature right after a support rebuild, is no longer accepted
  as self-consistent.
- The conformal dark U(1) model splines the potential along the bounce path with
  10000 instead of 1000 samples. With 1000 samples the bounce action was biased
  towards strong supercooling (for `y = 0.01`, `v = 0.14 GeV`, `g = 0.554`:
  `Tperc` 31 % too low, `alpha` 4.3 times too high), scattered along smooth
  parameter lines, and some strongly supercooled points failed. The new setting
  agrees with the unsplined potential to within 0.2 % in `Tperc`.

## [2.1.0] - 2026-07-28

Bugfix release: reheating temperature (Treh) is now computed from the
evolved true-vacuum energy density rather than assumed instantaneous.

### Fixed

- **Treh (reheating temperature)**: previously solved for Treh by assuming
  instantaneous reheating via energy conservation at the single percolation
  temperature (`eBRO(Treh) = eSYM(Tperc)`), discarding the gradual
  true-vacuum-energy-density evolution the percolation routine already
  tracks per step. This disagreed with the paper definition and overestimated
  Treh for deeply supercooled transitions (low Tperc). The `TBROint` spline
  built from the evolved `(TSYM, TBRO)` trajectory is now restored on
  `PercolationResult` and evaluated at `Tperc`, realigning the adaptive-step
  path with `transitionObservables_fixedstep.py`. The direct `Tb_criterion`
  solve remains as a fallback when the spline is unavailable, and
  `Treh = Tperc` as the final fallback.

### Added

- `TL_2HDM_BSMPT.py` and `TL_2HDM_BSMPT_highacc.py` model files, needed to
  use the `bp_lisa.yaml` benchmark point.

### Changed

- Installation check now shows a progress indicator while warming the numpy
  cache, so the install doesn't appear to hang; installation instructions
  updated to match.
- README and FAQ now reference CosmoTransitions and arXiv:2303.10171 for the
  LTE velocity approximation; `hydrodynamics.py` header updated accordingly.
- Issue and feature request templates updated.

## [2.0.1] - 2026-05-18

Compatibility and packaging fixes; no physics changes.

### Fixed

- **numpy ≥ 2.0 compatibility**: `np.product()` (removed in numpy 2.0) replaced
  with `np.prod()`. Percolation criterion functions in `bubbledynamics.py` and
  `bubbledynamics_fixedstep.py` now return Python scalars instead of shape-`(1,)`
  arrays, fixing a `SystemError` from `scipy.optimize.brentq` on numpy 2.x.
  `np.linalg.lstsq` calls updated from deprecated `rcond=-1` to `rcond=None`.
- **Windows compatibility**: `signal.SIGALRM` and `signal.alarm()` are POSIX-only
  and not available on Windows. All usages are now guarded by `hasattr` checks,
  fixing an `AttributeError` on import that caused the conda-forge Windows CI to
  fail.
- **PTA diagnostics**: when `ptarcade.signal_builder` fails to import, the
  underlying error is now shown alongside the "PTA likelihood unavailable" warning.

### Added

- `conda/meta.yaml` recipe for conda-forge submission.
- `build` and `twine` added to the `[dev]` optional dependencies in
  `pyproject.toml`.
- GitHub Actions workflow (`.github/workflows/publish.yml`) for automatic PyPI
  publishing on tagged releases via OIDC trusted publishing.

## [2.0.0] - 2026-05-06

Major code release accompanied by the paper
[arXiv:2605.15259](https://arxiv.org/abs/2605.15259).

### Added

- Public packaged interface (`pip install -e .`, `tl` console script,
  YAML-driven scans). v1 was a collection of top-level Python scripts
  (`My_point.py`, `My_point_comparison.py`, `My_potential_plot_point.py`,
  `My_scan_LISA_low_res.py`, `scanner.py`, `tl_dark_photon_model.py`)
  invoked individually; v2 ships a single CLI and a structured
  `transitionlistener` package under `src/`.
- Self-consistent percolation solver. The new
  `adaptive_step_size` algorithm iterates the Hubble rate and the
  false-vacuum fraction until convergence and is the default. A
  `fixed_step_size` static-grid solver is kept as a benchmarking
  baseline; both run from the same model files via the
  `percolation_algorithm_mode` YAML override (or `PercolationConf`
  default).
- ODE-based percolation integral (`percolation_integral_method = "ode"`,
  default) that uses the symmetric-phase sound speed and the integrated
  cosmological scale-factor ratio
  (`percolation_time_temperature_mode = "sound_speed"`, default) so the
  evolved `P_t(T)` already includes both effects without a separate
  `Pr_exp` channel. The historical bag/`c_s²=1/3` form is still
  reachable via `percolation_time_temperature_mode = "bag"`.
- Full thermodynamic alpha catalogue:
  - `alpha` (the canonical column), now defined as the
    Giese:2020 / pseudo-trace-anomaly definition
    (`alpha_thetabar`).
  - `alpha_theta` (bag-EOS) is still emitted as a separate output
    column so users can compare definitions on a per-point basis.
  - `alpha_p`, `alpha_e`, `alpha_hyd`, `alpha_inf`, `alpha_eq`.
- LTE bubble-wall velocity solver based on
  arXiv:2303.10171 (`Hydrodynamics.calcWallVelocityLTE`), reading the
  hydrodynamic coupling option from `pot.config.gwConf.coupled_hydrodynamics`.
  Free-input numeric `wall_velocity` values in `(0, 1]` are also
  accepted.
- LISA Cosmology Working Group spectrum recommendations
  (arXiv:2403.03723): bubble-collision + sound-wave + turbulence
  contributions with sound-wave-source lifetime.
- `precision_mode` bundles for tracing and tunnelling tolerances:
  `default`, `robust`, `xtrace`, `tunneltight`, `benchmark`.
- Output schema and observability:
  - Stable, fully columnar `output_table.csv` with documented columns
    (see paper App. A).
  - Documented error codes in `transitionlistener/errors.py`.
  - SNR computation against LISA, BBO, DECIGO, ET, muAres, and the
    PTA datasets (NANOGrav, EPTA, IPTA).
  - PTA log-likelihood evaluation through `PTArcade`.
  - UltraNest integration for nested-sampling scans of large parameter
    spaces.
- Multi-Higgs / multi-field potentials:
  - `models/TL_2HDM.py` and high-accuracy variants,
  - `models/TL_dark_flipflop.py` (two-singlet vev-flip-flop),
  - `models/TL_dark_U1.py`, `models/TL_dark_U1_g_parameterization.py`,
  - `models/TL_conformal_dark_u1.py`,
  - `models/templatePotential.py` as a starting point for new models.
- Automatic SM and BSM degrees-of-freedom accounting at finite T,
  including a kinetic-equilibrium switch
  (`pot.config.gwConf.coupled_hydrodynamics`) that controls whether
  the species coupled to the transitioning scalar and the rest of the
  heat bath are treated as a single hydrodynamic fluid.
- Self-consistent energy-density evaluation from the user-defined
  effective potential (replacing the older `ΔV` / bag approximation).
- Stable down to extreme supercooling: `α ~ 10^10` benchmarks pass.
- Reproducibility infrastructure for the paper:
  - `arxiv/figures/paper/configs/`: every YAML scan that feeds a
    paper figure.
  - `arxiv/reproducibility/paper/manifest.yaml`: figure → builder
    script → YAML inputs map.
  - `arxiv/reproducibility/paper/scripts/build_all.py`: walks the
    manifest, optionally re-runs the underlying tl scans, and rebuilds
    the figures (`--regenerate`, `--mode {adaptive_step_size,
    fixed_step_size}`, `--only LABEL`).
- Sphinx documentation: <https://tasillo.de/TransitionListener/>.
- `tests/test_release_smoke.py`: 1-2 minute end-to-end smoke that runs
  `tl -c examples/example_point.yaml` and asserts `Tperc`, `Treh`,
  `alpha`, `alpha_thetabar`, `RH` in their expected physical bands.
- GitHub Actions CI that installs the package (with the dev extras
  for pytest) and runs the smoke test on every push to `main` /
  `release`.
- `CITATION.cff` and an expanded `pyproject.toml` (project URLs,
  classifiers, dev extras).

### Changed

- `derived["alpha"]` now equals the Giese:2020 definition
  (`alpha_thetabar`) instead of the bag-EOS one. Old code that read
  `alpha` will continue to work but will pick up the new (more
  accurate) value; the old definition is still available as
  `alpha_theta`.
- The package layout moved from `TransitionListener/*.py` (top of
  repo, with hand-written paths in user scripts) to a proper
  `src/transitionlistener/*.py` import root with subpackages
  (`interface/`, `counterterms/`).
- `transitionFinder.py` (the v1 monolith) was split into:
  `phases.py` + `transitions.py` + `transitionObservables.py` +
  `bubbledynamics.py` + `nucleation.py`.
- The `xi` / temperature-ratio quantities (`xi_nuc`, `Tnuc_DS`
  vs. `Tnuc_SM`) are no longer first-class outputs (see Removed).

### Removed

- Computation of the temperature ratio between the transitioning
  sector and the external bath, and its propagation into the phase-
  transition observables. v1 carried `xi_nuc`, `Tnuc_DS`, `g_eff_DS`,
  `g_eff_SM` separately; v2 reports a single SM-temperature column
  (`*_SM_GeV`) and assumes thermal equilibrium between the two sectors
  during the transition (controlled by
  `pot.config.gwConf.coupled_hydrodynamics`). For models in which the
  two sectors decouple, set the flag to `False` and supply the
  decoupled-sector `geff` callbacks on the model.
- Computation of the dilution factor `D`. v1 shipped
  `TransitionListener/dilution.py` (a full Boltzmann solver inspired
  by arXiv:1811.03608 for non-thermal mediator decays). v2 leaves
  `D` as an open parameter on the GW spectrum (default `1`); users
  who need the dilution can compute it externally and pass it in.

## [1.0.2] - 2023-12-06

Added a conda environment file.

### Added

- `TL.yml`, a conda environment file intended to provide a working v1 setup
  with `scipy==1.10.1` and the small Python dependencies needed by the
  original scripts.

### Changed

- `My_scan_LISA_low_res.py` was wrapped in a standard
  `if __name__ == "__main__":` entry point so the example scan script behaves
  more cleanly when imported or launched from different environments.
- The README compatibility note was updated to document the `scipy`-related
  failure mode in the CosmoTransitions backend more precisely and to point
  users to the new conda environment file.

## [1.0.1] - 2023-10-12

Documentation-only maintenance update on the public v1 repository.

### Changed

- The README command examples were corrected to match the actual script names
  (`My_point.py`, `My_scan.py`, `My_comparison.py`).
- A compatibility note was added documenting that the v1 code path can run
  into CosmoTransitions error-code-7 failures on then-current Python / NumPy /
  SciPy stacks, especially on Apple M2 systems, while the authors could still
  confirm a working environment with `python 3.9.16`, `scipy 1.10.1`, and
  `numpy 1.24.3`.

## [1.0.0] - 2021-10-21

Initial public v1 release of TransitionListener on GitHub, corresponding to
the original dark sector workflow used in
[arXiv:2109.06208](https://arxiv.org/abs/2109.06208).

### Added

- A script-driven analysis workflow built on top of CosmoTransitions for
  dark sector first-order phase transitions and their stochastic
  gravitational wave signals.
- Single-point, grid-scan, comparison, and potential-cross-check entry points
  via `My_point.py`, `scanner.py`, `My_point_comparison.py`,
  `My_scan_LISA_low_res.py`, and `My_potential_plot_point.py`.
- Model support for a dark `U(1)` extension through `tl_dark_photon_model.py`
  and the bundled `TransitionListener/` module directory.
- Computation of nucleation temperature, transition strength, inverse
  timescale, relativistic degrees of freedom, entropy-dilution effects, GW
  spectra, and detector signal-to-noise ratios in the original v1 framework.
