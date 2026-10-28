# Changelog

All notable changes between releases of TransitionListener.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

Release 2.2.0 is a correctness release. Numbers from 2.1.0 move: the expansion history,
the pseudo-trace transition strengths, the redshift of the spectrum and the counting of
degrees of freedom were each computed in a way that did not match the definition they
claim, and two dozen defects could return a wrong number without saying so. Four
observables and six diagnostics are new, and the degrees-of-freedom scheme is now a
setting. Each entry below says what changed, by how much, and on which models; the full
account of every one is in its pull request.

### Added

- **Four verdicts the solver formed and never reported.**
  `WARNING:accuracy_targets_missed` and the three `DIAG:accuracy_*_uncertainty` columns say
  whether the accepted percolation profile met its `acc_tperc`, `acc_tfinal` and `acc_rh`
  targets, which are advisory, so this is the only record of whether they were honoured; a
  profile whose mean bubble separation `R*H` missed its 1 per cent target was previously
  indistinguishable from one that met it, with a 90th-percentile spread of 27.6 per cent.
  `WARNING:v_wall_no_solution` and `DIAG:v_wall_branch` say whether the wall-velocity search
  found no solution and which branch it chose.

- **Six diagnostics of how a point was reached**, as distinct from what the point is:
  whether an action was recomputed at a tighter path deformation
  (`WARNING:action_rescue_attempted`), whether that settled it
  (`WARNING:action_rescue_failed`), how many were recomputed (`DIAG:action_rescue_attempts`),
  how many image pairings were offered to the bounce solver and how many resolved
  (`pairings_tried`, `pairings_resolved`), and the point's own wall-clock time.

- **`entropy_definition`, which selects the thermodynamics of the expansion history**, either
  `"dof_table"` or `"eff_potential"`, with `constants.h_eff_today` beside it. A warning is
  raised at points where the choice matters, when the two schemes differ by more than 4 per
  cent. Over 91 points of four scan lines the estimated shift has a median ratio of 1.06 to
  the measured shift, so read it as a diagnostic of the expansion history and not as a
  prediction for an observable.

- **`WARNING:false_vacuum_not_shrinking`, raised where the physical volume of the false
  vacuum still grows at the percolation temperature**, the criterion of Ellis, Lewicki and No,
  arXiv:1809.08242. Both solvers raise it, and `falseVacuumVolumeGrowthRate` returns the growth
  rate in units of 3H.

- **Two diagnostics of whether the sound speed means anything.** `WARNING:noisy_c_s` and
  `DIAG:c_s_step_change` say whether the value survives a change of the derivative step by a
  factor ten in each direction, because a value can be round-off and still be a speed, so no
  test on the value itself can find it.

- **Two settings that were fixed numbers.** `betaH_S3_fit_rescue`, with the run-time override
  `percolation_betaH_S3_fit_rescue`, switches off the refit of an unstable action slope; and
  `counterterm_derivative_step`, default 1e-3 in internal field units, sets the finite-difference
  step of the zero-temperature one-loop derivatives that fix the counterterms.

- **`Treh_DS_GeV`, the reheating temperature of the transitioning sector**, beside
  `Treh_SM_GeV`, which is now the Standard Model bath's. The model attribute `SM_bath` names
  which radiation bath holds the Standard Model.

- **A refusal to complete now reports the percolation temperature it had already determined.**
  All 14 such points on the 2HDM marginal line had passed the percolation criterion before
  they were refused.

- **The bounce can be computed on every image of the false vacuum, behind
  `TracingConf.pair_transition_over_images`, which defaults to `False`.** Where the effective
  potential is invariant under a group, each minimum stands for a set of images of one
  physical state, and which image the phase tracing landed on is settled by the sign of a
  numerical step. `bounceActionOverImages` offers every image to the solver and keeps the
  smallest action. The default is off because the canonical-image fix below repairs the defect
  where it arises: over the forty couplings of the 2HDM `lambda3` line from 5.50 to 5.87,
  turning the setting on changes no observable at any coupling and raises the median runtime
  of a coupling from 408 s to 529 s. It is kept so the check can be run on a new model or a
  new symmetry group.

### Changed

- **One entropy for the expansion history.** The scale factor came from the counted degrees of
  freedom and the sound speed from the effective potential, which are two forms of the same
  relation. Both now come from the entropy `entropy_definition` selects, taken along the traced
  phase. Every adaptive run with the default `time_temperature_mode = "sound_speed"` changes.
  On the shipped example point, the conformal dark U(1) at g = 0.7 and v = 0.1 GeV, the shifts
  are at most 0.07 per cent. Over 95 points on four lines the median shift is 0.09 per cent in
  the percolation temperature `Tperc` and 2.4 per cent in the mean bubble separation on the
  2HDM line, reaching 3.9 per cent. A published number should say which `entropy_definition`
  produced it.

- **The pseudo-trace transition strengths divide both phases by the broken-phase sound speed**,
  as in arXiv:2004.06995 eq. (2.13). Each phase was divided by its own, which left whatever the
  two share in the difference. `alpha` and `alpha_hyd` change in both solvers, `alpha_thetabar`
  in the adaptive one and `alpha_theta` in the fixed step size one; `Tperc`, the reheating
  temperature `Treh`, the inverse duration `beta/H` and the peak frequency do not, the
  amplitude does.

- **The spectrum is redshifted from the right temperature.** `Treh_SM_GeV` is the Standard
  Model bath's temperature at reheating rather than the transitioning sector's, and
  `g_eff_tot_reh` and `h_eff_tot_reh` are evaluated there instead of at `Tperc`. Spectra and
  signal-to-noise ratios change by a fraction of a per cent where the degrees of freedom barely
  move between the two temperatures, and by up to a factor 1.36 in peak frequency and 0.75 in
  amplitude on the strongly supercooled conformal dark U(1) line, where `Tperc` is 40 keV and
  `Treh` 10.6 MeV.

- **Standard Model fields declared inside a potential are counted.** The 2HDM models put the W
  and Z bosons, the top quark, the Higgs and others in the potential and flag them, and they
  were counted nowhere, so the effective number of entropy degrees of freedom at reheating
  `g_eff_tot_reh` came out near 69 instead of near 98. The 2HDM spectra move by about +6 per
  cent in peak frequency and -10 per cent in amplitude. Models with no Standard Model fields in
  the potential are unaffected.

- **Today's entropy degrees of freedom are 3.9309** (arXiv:1803.01038) and were 3.91. All
  amplitudes rise by 0.7 per cent.

- **The hybrid wall-velocity limit is found by searching its range** instead of returning the
  search bound where the matching residual has not crossed zero. A wall that should run away
  was computed as a hybrid at two couplings, `v_wall` 0.7475 and 0.9509 instead of 1, taking
  the LISA signal-to-noise ratio at one of them from 0.8076 to 11.269. 125 of 127 couplings on
  the 2HDM line are bit-identical. This departs from the reference implementation of
  arXiv:2303.10171, from which the routine is taken verbatim.

- **A model's declared path-deformation target reaches the solver.** A sentinel was read too
  late, so a model's own `fRatioConv` was overwritten by the default.
  `TL_2HDM_BSMPT_highacc` declares 5e-3 and every run of it had used 1e-2; at the 2HDM edge
  point the two give `S_3/T` of 211.518 and 212.086, where `S_3` is the Euclidean action of the
  O(3) bounce. Results from that model variant move. Models that declare nothing are
  unaffected.

- **The bubble wall is driven by a strength normalised to the transitioning sector alone.**
  `calcAlphas` returns `alpha_hyd_wall` beside `alpha_hyd`, and the equilibrium Lorentz factor
  `gamma_eq` and the bubble-collision efficiency `kappa_col` take the wall strength rather than
  the strength that drives the sound waves. The fixed step size enthalpy is referenced to the
  zero-temperature vacuum energy. `kappa_phi` changes where the wall runs away.

- **The temperature-derivative step of the transition strengths is sized on what is being
  differenced**, through `helper_functions.temperatureDerivativeStep`, which balances the
  round-off of differencing the vacuum offset of the potential against the stencil's truncation
  error and so depends on the field value. It was a fixed fraction of the temperature.

- **The counterterm derivative step no longer follows the phase-tracing tolerance.** The
  zero-temperature one-loop derivatives at the vacuum expectation value that fix the
  counterterms took their step from `x_eps`, so tightening a tracing tolerance changed the
  potential. They now take `counterterm_derivative_step`, whose default 1e-3 is the step they
  have always used in the default setup. On the abelian dark Higgs, where the Coleman-Weinberg
  term at the vacuum is of order 1e9 in internal units, steps below 1e-3 are dominated by
  round-off: 1e-4 shifts the second derivative by up to 1.4 per cent.

- **An unstable action slope is retried at a tighter path deformation.** `calc_betaH_S3` fits
  the slope of `S_3/T` near the percolation temperature twice, over the support samples and
  over a subset; where the two disagreed by more than `betaH_S3_fit_rel_tol` the result was
  reported through `WARNING:betaH_S3_fit_unstable` and left alone. At the default deformation
  the actions are the reason: on a 2HDM benchmark `S_3/T` scatters by 0.5 about a straight line
  across the fitting window against a physical variation of 0.2, so the slope and its sign are
  undetermined. The slope is now refitted on a stencil of its own with the deformation
  tightened, which turns -27.3 into +66.5 on that benchmark, against +67.0 to +67.3 from
  fitting resolved actions.

- **`percolation_maxit` is 20 and was 10.** Every point rejected for an unresolved percolation
  probability had unspent action budget. About 2 per cent more action solves, and a fixed step
  size run that previously stopped at iteration 10 can return a different `Tperc`.

- **A rescued action is recomputed with real margin** over whatever the run configured, rather
  than clamped to a target the model may already declare, which made the rescue recompute an
  outlier with the settings that produced it. Halving the deformation target took the action
  scatter from 0.78 to 0.28 rms in `S_3/T` on the 2HDM edge line.

- **Two routines are faster and return the same numbers.** The bounce integrator's step-size
  error estimate no longer builds a list of arrays, which is 16 per cent off a bounce with
  actions identical digit for digit; and the path-deformation basis grows when the force ratio
  plateaus above its target instead of stopping there, which is about 2.5 times faster on the
  2HDM edge line.

- **Two changes decide which samples are flagged, not which numbers are returned.** The
  action-rate jitter gate no longer judges a bubble nucleation rate that has died: samples more
  than 30 decades below the in-band rate peak are excluded, with liveness taken from the
  cold-side running maximum. And a retried action starts from the straight line between the
  minima, as every other temperature does.

- **Internal, with no effect on results.** The error codes a point can return, the tests, and
  the documentation of `logGamma` and of two docstrings that rendered wrongly in the published
  pages. A rejected profile now says which resource ran out, and the diagnostic plot of the
  degrees of freedom draws what it claims to.

### Fixed

Each line says what was wrong and when it bit.

- A broken-phase temperature below the symmetric one now fails the point with an error code.
  The transition releases energy and happens below the critical temperature, so the broken
  phase is reheated and `T_bro > T_sym` holds by construction, whatever accuracy the solver
  works to. The old guard raised a bare `ValueError`, which carries no code, so the CLI
  reported an unclassified failure, code -1. It is now an assertion of that ordering whose
  only slack is the floating-point grid: a relative shortfall below 1e-9 is clamped with a
  `RuntimeWarning`, and anything larger is refused as a `PercolationError`, code 10, naming
  both phases and the relative latent heat at `T_sym`. The threshold is measured: over 2313
  step-3 support points on the 2HDM `lambda3` line the worst negative latent heat among
  correctly ordered points is -5.91e-14, while the one reversed point, `lambda4` =
  -2.0807692307692305, carries -1.264e-04, which the default `acc_tperc` would have admitted
  silently. A non-finite temperature now raises rather than being returned, and the rule,
  previously in three copies, lives in one shared function.
- No failing point is reported as error -1 any more, and every one carries its cause. An
  exception without an `errorcode` now reports `UnexpectedError`, code 999, with its original
  type named. `errormsg` is a registered result column that no failing path filled, the
  per-point handler leaving it empty and the random-scan worker blanking it after reading it;
  both now carry it. A scan writes the cause beside its numeric columns in `errormsg.txt`, one
  row per failed point with its grid position, for both formats, since neither the float arrays
  nor the hdf5 datasets hold a string column. The file is absent when no point failed, and one
  left by an earlier run over the same output path is removed rather than read as describing
  the new results. Each field stays one row: whitespace runs collapse to a space, and on the
  random-scan path commas become semicolons, because that table is comma-separated and quotes
  nothing.
- The transition could be built between two images of the same minimum, so the bounce was
  computed along a path the field does not take. On the 2HDM at `lambda3` = 5.868 that gave
  `S_3/T` = 336.6 against 152.6 and 153.3 at the neighbouring couplings, and the point returned
  no percolation.
- The canonical image of a traced phase was decided by the sign of a field component the tracer
  does not resolve: two nodes 2e-09 GeV apart were put on images 492 GeV apart. And a phase that
  needed no transformation at all could be mirrored, because the search for the transformation
  took the first of several equal distances, and at the origin every element of the group ties
  with the identity.
- A phase that is its own mirror image was duplicated, and the duplicate entered the transition
  graph.
- The tracer followed a point after it had stopped being a minimum. On the 2HDM at
  `lambda3` = 5.813 that left 7 of 61 nodes of the high-temperature phase at the origin between
  T = 8.5 and 39.0 GeV, where the smallest curvature reaches -377.8 GeV^2, and the bounce was
  computed from there.
- A phase could leave the range of its own traced minima between two traced points, because the
  interpolating cubic overshoots.
- A bounce whose two bracketing trials disagreed on the action returned the last trial's number
  anyway. It is reported as unresolved now, which is what it was.
- The shooting bisection converged on an absolute tolerance in its own variable, so it could
  stall without converging, and a trial with no usable slope was taken as a verdict.
- The bounce profile was integrated and judged to have arrived on a scale unrelated to the
  bounce's own.
- The bounce action died where the barrier is small compared with the path, which is the
  strongly supercooled corner. On the conformal dark U(1) ladder the lowest percolating coupling
  moves from g = 0.54 to g = 0.42, twelve couplings gain a percolation, and `Tperc` is unchanged
  to between 1.8e-06 and 2.4e-04 per cent where the old code resolved it. This is the largest
  single change in the release.
- The percolation estimate took a root of its criterion that need not be the hottest one, and
  returned 1.97e-12 GeV where the answer was 61.5 GeV.
- The check that the transition chain ended in the zero-temperature minimum refused chains
  that had. It compared field values component by component, which gives the smaller field
  component a tolerance tighter in proportion while the tracer's error is the same absolute size
  in both: on the 2HDM at `lambda3` = 5.8672 the endpoint sits 2.7e-06 of the field scale from
  the minimum, with the potential agreeing to eight digits, and it was refused on the smaller
  component alone. The check now compares the depth of the potential at the two points.
- The adaptive percolation controller spent its budget below the transition and refused to
  repair a jump it then rejected the profile for: eight faults, found by instrumenting a
  40-point 2HDM edge line on which 31 of 40 points resolved.
- Energy densities and the potential did not broadcast over an array of temperatures, a figure
  saved to a path without an extension lost the default format, the summed spectrum lost its
  hatching, and a spectrum plot given only an axis could not find its figure.
- The percolation integral was evaluated twice on coarse support grids.
- The bubble nucleation rate was allowed to grow above the temperatures whose action supported
  it.
- Two temperatures were compared against an absolute floor of one rather than against their own
  size, which swallowed every refinement point for a transition below one in the model's
  internal units.
- The daisy term was evaluated in a form that threw away its significant digits.
- The wall velocity took its sound speed from the potential alone, not from the thermal part
  too, and a sound speed at or above one was replaced rather than refused.
- The scale factor of the percolation integral, the expansion history of the mean bubble
  separation, and the time-temperature factor of `(beta/H)_S3` each used a thermodynamics that
  did not match the one the rest of the integral assumed.
- The reheating temperature was read off a quantity that carried scatter, since 2.1.0, and
  `Tperc` in the adaptive solver was not taken from the solver's own event.
- The degrees of freedom of the fields of the potential, the Standard Model fields among them,
  and the decoupled radiation baths were each counted wrongly or not at all.
- The transverse photon mass in `models/TL_2HDM.py` was wrong.
- The break frequencies of the summed spectrum were redshifted by the wrong power.
- The conformal dark U(1) model splined the potential along the bounce path in a way that lost
  accuracy where it mattered.
- A point that timed out came back carrying numbers rather than a timeout, and a timeout raised
  inside the initial-condition search was read as a wall too thin.
- A degenerate third-step profile reached percolation with nothing evolved.
- A broken phase with no thermal pressure, and the symmetric phase at extreme supercooling,
  were both mishandled.
- The profile plot drew the false vacuum on the image the phase tracing produced rather than on
  the image the bounce was computed on.
- `examples/bp_lisa.yaml` named a model file that is not in the repository, so the shipped
  benchmark point could not be run.
- A failure of the potential whose tag the bounce code does not handle raised `IndexError`
  rather than reaching the caller as `UnexpectedError`, because the message was read from an
  index of `args` that `PotentialError` never fills. The second copy of the same translation,
  in `bounceAction`, reported no tag at all.
- The tachyonic-mass message divided the squared boson masses by the internal field scale,
  which is neither a mass nor a squared mass and changes with a model's choice of scale, and
  printed every mode without saying which one was tachyonic.
- `Phases.testTmax` documented `ValueError` and raises `WrongHighTPhaseError`, and
  `PotentialError`'s docstring described a tuple convention its constructor does not follow, so
  a reader deciding what to catch would have caught nothing.

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
