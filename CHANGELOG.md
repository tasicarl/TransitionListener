# Changelog

All notable changes between releases of TransitionListener.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Four verdicts the solver formed but never reported are now observables.**
  `WARNING:accuracy_targets_missed` and the three `DIAG:accuracy_*_uncertainty` values say
  whether the accepted percolation profile met `acc_tperc`, `acc_tfinal` and `acc_rh`. Those
  targets are advisory -- the controller accepts either way -- so this is the only record of
  whether they were honoured; a profile whose `RH` missed its 1% target was previously
  indistinguishable from one that met it, with a 90th-percentile spread of 27.6%. The warning is
  raised only where an uncertainty was measured and exceeded its target, or a required
  observable could not be determined at all. False with finite uncertainties means every target
  that could be measured was met, including the common case where `acc_rh` could not be judged
  because no second estimate of the mean bubble separation existed to difference against, which
  `DIAG:accuracy_rh_uncertainty` reports as `inf`; false with `nan` means no verdict
  was formed at all, which is the fixed step size backend and the adaptive controller's
  iteration-limit exit. `WARNING:v_wall_no_solution` fires where `find_vw` returns exactly
  0, which is a verdict that no wall solution exists but which zeroes every kinetic energy
  fraction, so `LISA_SNR` comes out exactly 0.0 and reads as a prediction of no signal; note
  that `v_wall == 1` is the *normal* outcome on a supercooled line, so only the exact zero
  is a non-solution. `WARNING:v_wall_branch_ambiguous` fires where the wall-velocity residual
  changes sign more than once between adjacent samples of the search grid, i.e. where the
  branch was chosen rather than determined.

- **A refusal to complete reports the percolation temperature it had already determined.**
  All 14 such points on the 2HDM marginal line, the 29-point zoom at the no-percolation
  edge, reached `Pint` between 0.3215 and 0.7640, above `f_perc = 0.28957`, so `Tperc` was
  in hand when the point was refused.

- Four diagnostics of how a point was reached, rather than of what it is.
  `WARNING:action_rescue_attempted` says that actions were recomputed with the path
  deformation tightened, by either the rate-jitter rescue or the fit rescue above;
  `WARNING:action_rescue_failed` says that this did not settle the quantity that
  prompted it. `DIAG:action_rescue_attempts` counts the actions recomputed and
  `DIAG:runtime_s` gives the wall clock of the point, which is what makes the cost of a
  rescue visible: a rescued 2HDM point took 1887 s against about 775 s without one.
  The rate-jitter rescue already recorded its outcome internally and nothing read it.
  These are written to `1_All_params.txt` under their own `Diagnostics:` heading, apart
  from the observables, so that two runs of one point still agree there byte for byte.


- Output column `Treh_DS_GeV`: the reheating temperature of the
  transitioning sector, i.e. the fields of the potential and the coupled
  radiation bath.
- Model attribute `SM_bath` (`"coupled"` or `"decoupled"`) that names the
  radiation bath holding the Standard Model. If it is not set, TL takes the
  bath whose tables are `e_geffSM`, and warns if both are. A model with
  Standard Model fields in its potential (e.g. the 2HDM) cannot decouple the
  Standard Model; TL raises an error when it is set up that way.
- `constants.h_eff_today`, today's entropy degrees of freedom.
- Setting `entropy_definition` of the percolation solver, `"dof_table"` (default)
  or `"eff_potential"`, with the run-time override
  `percolation_entropy_definition`. It selects the thermodynamics of the
  expansion history. Both describe the same plasma, the fields of the potential
  and the coupled radiation bath without the decoupled one, and differ in how they
  treat the fields. `"dof_table"` counts them as free particles at their
  zero-temperature masses and adds the tabulated entropy degrees of freedom of the
  coupled bath. `"eff_potential"` takes `-dV/dT` of the effective potential, which
  gives the fields their thermal masses and the Arnold-Espinosa daisy term; it
  carries the tabulated bath with its perturbative corrections too, since the
  field-independent part of the potential is built from the same tables, and what
  it lacks is those corrections for the fields of the potential themselves.
  Neither scheme contains the other: the counted one has the perturbative
  corrections everywhere and no thermal masses, the potential one has the thermal
  masses and the daisy term but free-particle counting for the fields. No
  published work decides between them, so the difference is a modelling
  uncertainty that can now be measured by running both.

- **A flag for points where the entropy scheme matters.** `entropy_definition` chooses
  between two approximations to one entropy density, and neither contains the other, so the
  choice is a modelling uncertainty of the expansion history rather than a numerical one.
  Every adaptive step size run in `sound_speed` mode now evaluates the other scheme on the
  support that fixed the percolation temperature and reports `DIAG:entropy_scheme_cs2_spread`, the median relative
  difference
  between the two schemes' `3 c_s^2` over that support, and `DIAG:entropy_scheme_lna_gap`,
  the difference of their logarithmic entropy ratios over three.
  `WARNING:entropy_scheme_sensitive` is raised when the spread exceeds
  `percolation_entropy_scheme_warn_threshold`, 4% by default, and the comparison can be
  switched off with `percolation_entropy_scheme_diagnostic`. The fixed step size solver does
  not read `entropy_definition` and refuses anything but the default, so it has no second
  scheme to compare against: it writes the flag as false and both diagnostics as `nan`. It costs one further pass over the
  percolation support to build the other scheme's history, which is three entropy
  evaluations per support point, at the temperature and at both endpoints of the derivative
  stencil, and no further bounce actions. Measured on the example point, the runtime was
  121.3 s against 121.6 s without the comparison, and `Tperc` and `R_*` are unchanged.

  The median spread is an estimate of how far the mean bubble separation of that point moves
  between the two schemes. Over 91 points of four scan lines, each run in both schemes so
  that the shift is known, the ratio of the estimate to the measured shift has median 1.06
  and lies between 0.88 and 1.29 for eight points in ten, with 93% of points inside a factor
  of two; the bias is between 1.00 and 1.17 on every line separately.

  The default threshold says "flag a point whose `R_*` is uncertain at about the five per
  cent level from this choice". It sits a little below five so that the scatter of the
  estimate does not hide such a point, and the precise value is not critical. On the
  calibration sample four per cent flags one of the ninety-one, whose separation moves by
  3.7%; the largest measured shift in that sample is 3.8%, so the sample constrains the
  estimate rather than the threshold. A run that wants the weaker cases can lower it: at 3.5%
  the sample flags 6 points, every one of which moves by more than 2.9%, and none that moves
  by less than 2%.

  Two choices behind those numbers are worth recording. The statistic is the median over the
  support and not the largest value, because the shift integrates the difference over the
  whole history: the maximum overstates it by a factor 1.8 and a single support temperature
  that has lost a usable sound speed in one scheme drives it to 100% where the separation
  moves by a few per cent. And `DIAG:entropy_scheme_lna_gap` is reported but is not the
  criterion: it is the linearised estimate of the same shift, and its ratio to the measured
  shift has median 0.08 on an abelian dark Higgs line, 1.0 on a conformal dark U(1) line and
  5.2 on a 2HDM line, reaching 109% estimated against 3.8% measured at one 2HDM point. Read
  it as a diagnostic of the expansion history, not as a prediction for an observable.

### Changed

- **The action-rate jitter gate no longer judges a rate that has died.** It defines its band
  by probability, `0.01 <= P <= 0.9`, but measures the smoothness of the rate, and under
  strong supercooling `P` climbs past `f_perc` while `log10(Gamma/H^4)` falls through hundreds
  of decades. One quadratic in `ln T` cannot describe that: with the exemption disabled a
  perfectly smooth synthetic band is flagged with a 39.8-decade residual. All 22 rejections
  measured across the coarse, refined and marginal 2HDM lines sat 40.5 to 688.5 decades below
  the in-band rate peak, and 16 lay past percolation, where they cannot move `Tperc`, `RH` or
  `alpha`. Samples more than 30 decades below the peak are now excluded, with liveness
  taken from the cold-side running maximum so that a downward spike cannot exempt itself.

- **`percolation_maxit` is raised from 10 to 20.** Every point rejected for an unresolved
  `Delta P` had reached the support-update limit with 112 to 119 action evaluations unspent.
  This closes that class at about 2% more action solves. The setting is shared with the
  fixed-step backend, where each iteration recomputes the whole profile and no `n_action_max`
  bounds it, so a non-converging fixed-step run can spend up to twice the action solves and one
  that previously stopped at iteration 10 can return a different `Tperc`.

- **The hybrid wall-velocity limit is found by searching its range.** `Hydrodynamics.max_al`
  returned its *search bound* when the matching residual had not crossed zero by the top of the
  range, which is correct only while that residual is monotonic. Where it is not, the bound
  entered `alN > max_al` as though it were a physical strength, the runaway branch was skipped,
  and a wall that should run away was computed as a hybrid: `v_wall` came back 0.7475 and
  0.9509 instead of 1 at two couplings, taking the LISA SNR at the first of them from 0.8076
  to 11.269, a factor 14.
  The crossing is now sought across the range. **This departs from the reference implementation
  of `arxiv:2303.10171`, from which the routine is taken verbatim**, rather than fixing a defect
  in TransitionListener. Where the residual is monotonic the answer is unchanged, and 125 of 127
  couplings on the 2HDM line are bit-identical. Where it never goes negative there is no subsonic
  branch to bound at all: the old code then handed `root_scalar` a bracket with equal signs at
  both ends and raised a bare `ValueError: f(a) and f(b) must have different signs`, which
  nothing upstream types, so the point was reported as error -1. Sweeping 15 x 15 x 11 triples
  with `c_b^2` and `c_s^2` in [0.05, 1/3] and `psi_N` in [0.9, 1.1], 895 of the 2475 have a
  finite non-negative residual at both ends of the old bracket, the condition that made
  `root_scalar` raise, 227 of them with `psi_N` below one; the count depends on the `psi_N`
  window, which is why it is quoted. The search bound is reported there instead, which leaves the
  subsonic-against-runaway verdict to `find_vw`'s own `shooting(v_J) < 0` test.
  A strength at which the residual cannot be evaluated leaves a hole in the grid, which the
  search skips; the run's own `Timeout` is re-raised rather than absorbed into one, so a point
  whose deadline expires inside the search is reported as a timeout instead of sampling on
  past it. The sign changes behind `WARNING:v_wall_branch_ambiguous` are counted between
  adjacent samples only, so a hole cannot read as a crossing that the bracketed search itself
  would refuse. That count reaches nothing but the flag, and over 2475 triples swept with
  `c_b^2`, `c_s^2` in [0.05, 1/3] and `psi_N` in [0.9, 1.1], fifteen, fifteen and eleven
  values, the limit is bit-identical under either counting and the flag, raised on 105 of
  them, changes on none: the two differ only where a hole falls on a crossing, and no triple
  in that sweep leaves a hole on a grid that has one.

- **The path-deformation basis grows when the force ratio plateaus** above its target instead
  of stopping there, about 2.5x faster on the 2HDM edge line.

- **A retried action starts from the straight line between the minima**, as every other
  temperature does. Seeding a retry from a neighbouring temperature's converged path was
  measured to return a non-finite action on the 2HDM, so the retry would have voided the
  sample it was meant to rescue; the idea is not part of the solver. The rate-jitter retry
  still tightens the path deformation relative to what the run configured, as above.

- **A model's declared path-deformation target now reaches the bounce solver.**
  `generic_potential.__init__` tightens `fRatioConv` from the 1d default of 2e-2 to 1e-2 for
  multi-field models, and guards that override with a sentinel so a value the model chose is
  respected. The sentinel was read *after* `setConfigParameters()`, so it already held the
  model's own value, the equality test always passed, and the model's choice was overwritten.
  `models/TL_2HDM_BSMPT_highacc.py` declares `fRatioConv = 5e-3` and every run of it had been
  using 1e-2. The two settings are not equivalent: at the 2HDM edge line
  (lambda3 = 5.8164076664, T = 26.7116 GeV) they give `S_3/T` of 211.518 and 212.086, a
  difference of 0.247 orders of magnitude in the nucleation rate, so results from that model
  variant move. Models that make no choice of their own are unaffected and still get 1e-2.

- **The rate-jitter rescue always has real margin over what the run configured**, instead
  of only clamping to a fixed target a model may already declare. The clamp was
  `min(current, 5e-3)` for `fRatioConv` and `min(current, 1.0)` for `converge_0`, which are
  exactly the values `R2HDM_HighAcc_NAct100_Conv1_FRatio005` declares: every deformation
  parameter of the "tight" set was then bit-identical to the run's and the rescue recomputed
  the outlier action with the settings that produced it. `fRatioConv` is now divided by a
  further factor 5 after the historical cap of 5e-3, so there is always real margin and a
  run that already asks for something stricter still keeps it. Measured against the
  configured value, the tight target is 1e-3 for any run at or above 5e-3 -- a factor 10
  below the multi-field default of 1e-2 and a factor 20 below the 1d default of 2e-2 -- and
  one fifth of the configured value below that. `converge_0` keeps the plain
  `min(current, 1.0)` clamp, so the margin comes from `fRatioConv` alone. Halving
  `fRatioConv` from 1e-2 to 5e-3 took the action jitter from 0.78 to 0.28 rms in `S_3/T` on
  the 2HDM edge line, and the cost is bounded because the rescue only recomputes the handful
  of outlier temperatures it targets.

- **One entropy for the expansion history**: the time-temperature relation took
  the scale factor from the counted degrees of freedom and the sound speed from
  the effective potential. These are the same relation, since
  `1/(3 c_s^2) = 1 + (1/3) d ln g_*s/d ln T`: a constant `g_*s`, `c_s^2 = 1/3`
  and `a ~ 1/T` are one assumption and not three. Both now come from the entropy
  selected by `entropy_definition`, and both are taken along the traced phase: the
  sound speed is the logarithmic derivative of that same entropy, so it carries the
  `d2V/dXdT * dX/dT` term that a derivative at fixed field value drops. That term is
  exactly zero for a phase pinned at the origin and reaches 0.09% of `3 c_s^2` at a
  drift of `dX/dT = 0.5` on the 2HDM, so it matters only where the symmetric phase
  moves with temperature. `Tperc`, the mean bubble separation and
  `(beta/H)_S3` all belong to one equation of state. How much the choice is worth
  is a question for the first two of those: `(beta/H)_S3` is the slope of the
  action, and at the default path deformation that slope is limited by the scatter
  of `S3/T` rather than by the thermodynamics, on the 2HDM line in particular. The
  default, `"dof_table"`, stays defined below about 2 MeV, where after neutrino
  decoupling no single pressure with `dp/dT = s` exists and the potential route has
  nothing to refer to. How far apart the two are is a property of the model. As a
  fixed-field probe of the symmetric-phase thermodynamics at the 2HDM point of
  `tests/test_potential_broadcasting.py` (lambda1 = 0.006, lambda2 = 0.25,
  lambda3 = 8.27, lambda4 = -2.55, lambda5 = 0.76, m12^2 = 14186.7 GeV^2,
  tan_beta = 17.7, v = 246.22 GeV), with the fields held at the origin: over
  T = 160 to 31.2 GeV, which is the range in which that phase is traced, the counted
  route gives `3 c_s^2` between 0.977 and 0.997 against 0.955 to 1.037 from the
  potential, whose sound speed follows the thermal masses of the modes that are
  becoming heavy, a separation of 5.5%. Continued to 20 GeV, below the traced range,
  the potential route falls to 0.765 and the separation reaches 24%, but that is
  extrapolation: the phase is traced only down to 31.11 GeV and this point does not
  percolate at all. The window is part of the statement.

  **This moves existing results.** The default sound speed previously came from the
  potential while the scale factor came from the counted degrees of freedom, so
  every adaptive run with the default `time_temperature_mode = "sound_speed"`
  changes. On the released example point of `examples/example_point.yaml`
  (conformal dark U(1), g = 0.7, v = 0.1 GeV, y = 0.01) the shifts are small,
  because that model is close to conformal and the two entropies nearly agree:

  | observable | before | after | change |
  | --- | --- | --- | --- |
  | `Tperc_SM_GeV` | 0.0022119 | 0.0022120 | +0.005% |
  | `alpha` | 387.10 | 387.04 | -0.017% |
  | `RH` | 0.070683 | 0.070726 | +0.061% |
  | `betaH_RH` | 62.640 | 62.601 | -0.062% |
  | `betaH_S3` | 63.232 | 63.190 | -0.066% |

  Models whose `g_*s` varies strongly across the transition move much more. Over 95
  points on four lines (a 2HDM `lambda_3` scan, an abelian dark Higgs `lambda` scan, a
  conformal dark U(1) `g` scan and nine single benchmarks), the median shift against
  the previous mixed default is 0.09% in `Tperc` and 2.4% in `R_*` on the 2HDM line,
  reaching 3.9%, while the abelian line moves by 0.14% in `R_*`. `Tperc` is barely
  affected throughout; what moves is the mean bubble separation and with it
  `(beta/H)_RH`. Anyone reproducing a published number should say which entropy
  definition produced it.

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

- **A traced phase is stored on one image of the potential's symmetry.** Below the
  temperature at which the origin tilts over, the displaced minima are exactly degenerate
  images of one another, so which one a trace walks into is settled by the sign of a
  numerical step and not by the potential. Two seeds that reach the same physical phase on
  opposite images give two phases that `removeRedundantPhases` cannot tell are one, because
  it compares field values, and the transition can then be built from a high phase on one
  image to a low phase on the other, over a path the potential does not have. On the 2HDM
  at `lambda3` = 5.868 the high phase sat at `phi` = (-27.416, -15.335) GeV against
  (+27.354, +15.321) GeV and (+27.479, +15.349) GeV at the neighbouring couplings, with the
  broken minimum at (+11.48, +246.33) GeV in all three: a field separation of 264.54 GeV
  against 231.55 GeV, and `S_3/T` = 336.6 against 152.6 and 153.3 at T = 30.0 GeV. The
  smallest `S_3/T` over the cache was 246.4 against a nucleation criterion near 155, so the
  rate never rose, the percolation probability stayed at zero over all 197 accepted support
  points, and the point returned error 8 between two couplings that behaved. The
  transformation is now read off the node of the trace furthest from the origin, where
  every field component has a resolved sign, and applied to the whole phase, to `X` and
  `dXdT` alike; one transformation for the trace rather than one per point, which would
  send neighbouring nodes of a smooth phase to different images.

- **The tracer stops where the point it carries is no longer a minimum, and steps off a
  saddle when seeding a phase.** The test in `traceMinimum` was written as
  `negeig is True` against a `numpy.bool_`, which is never the `True` singleton, so it
  never fired and the trace walked on through temperatures where its point was a saddle.
  On the 2HDM at `lambda3` = 5.813 that left 7 of the 61 nodes of the high-temperature
  phase at the origin between T = 8.5 and 39.0 GeV, where the smallest eigenvalue of the
  Hessian runs down to -377.8 GeV^2, and the bounce was computed from a false vacuum that
  is a saddle. `forbidPhaseCrit` was compared the same way and is now read by value as
  well; the criterion shipped with the dark U(1) models returns a `numpy.bool_`, so it had
  been skipping nothing. A minimiser cannot leave a stationary point that symmetry holds
  the gradient at, so `descendFromSaddle` steps off it along the most negative curvature
  direction, in both signs, and returns the nearest minimum: the origin tilting over below
  some temperature is the generic behaviour of a thermal mass term with
  `m_eff^2(T) = -|m^2| + c T^2`, and in a strongly supercooled transition the universe
  reaches that temperature while still in the high-temperature phase, so the displaced
  minimum is the false vacuum the bounce has to be computed from. Phases the tracer
  stopped and restarted either side of such a merge are put back together by
  `mergePhasesMeetingAtAPoint`, which joins pieces separated by no more than 1e-3 of the
  traced span whose endpoints across the gap agree to better than `diftol`; the redundancy
  pass that follows works on phases overlapping in temperature and cannot see a pair
  separated by a gap. The computable corner of the 2HDM benchmark extends from
  `lambda3` = 5.818 to 5.866 on these changes, 19 couplings gained and none lost.

- **A bounce whose bracketing trials disagree on the action is reported as unresolved.**
  `bounceAction` returned whatever its last bracketing trial gave, so a solve that had not
  converged still produced a number, and that number is not the action of any bounce. Such
  a sample is now reported as unresolved and dropped on its own rather than the whole
  coupling being abandoned, and the percolation integral is built from the samples that did
  resolve. A bracketing trial is also recorded only from a trial that was integrated:
  the previous code could store a previous iteration's profile against the current trial's
  radius, or reach an unbound `r0` and raise.

- **A shooting trial without a usable slope is rejected; the solve carries on.** Before
  integrating a trial, `findProfile` asks `dV_from_absMin` for the slope at
  `phi(0) = phi_absMin + delta_phi0`, and it refused the whole solve on `dV < 1e-200`,
  which compares a signed quantity against a magnitude floor. The test was written for a
  thin-walled bubble, where `delta_phi0` is small enough that `dV` underflows, and it fired
  on every negative `dV` as well. A negative one is what the path's own potential gives
  near the true vacuum, where `SplinePath.dV` differentiates a cubic spline through sampled
  `V` whose ripple is large compared with a derivative that vanishes there: on the 2HDM at
  `lambda3` = 5.8191375291 with `V_spline_samples` = 1000, the spline returns -23.5875 at
  the path's true-vacuum endpoint at T = 60 GeV, -196.002 at 58 GeV and +7.79409 at
  55 GeV, against an analytic `grad V . dpath/dx` of +0.0117756, +0.0332661 and -0.0109657.
  At the other endpoint the same spline is right to between 0.02 and 1.0 per cent.
  `dV_from_absMin` covers the neighbourhood of the minimum with a Taylor form out to
  `phi_eps` = 2.46e-04, which is `1e-06` of the gap between the minima, while the spline's
  derivative becomes usable only once `d2V delta_phi0` clears that error floor, about five
  times further out at this benchmark. Between the two a bisection trial aborted the solve
  and `_find_profile` reported the result as an infinite action. Whether a given
  temperature died depended on whether a trial landed in that window, so the band had
  sharp, non-monotonic edges: at
  that coupling the action came back at 61.54, 59, 57, 55 and 50 GeV and was absent at
  61.52, 61, 60, 58 and 56.

  The slope is now read away from `phi_absMin`, as `dV * sign(delta_phi0)`, and a trial
  without one is rejected the way an over- or undershoot is: it bounds `x` from above when
  `phi0` is nearer `phi_absMin` and from below when it is nearer `phi_metaMin`. Finiteness
  is tested on its own and not left to the floor: a `nan` slope fails the comparison
  already, but only because every comparison against a `nan` is false, while an infinite
  one passes any floor and carried into `initialConditions`, whose `phi0` and `dphi0` then
  reached `integrateProfile` as the starting values of an integration that cannot mean
  anything. A solve in
  which no trial ever had a usable slope still reports `TooSmalldVError`, and so does one
  whose `x` stops being a number. The profile returned is built from the trial it was
  integrated from throughout, including the points interior to `R[0]`, which `exactSolution`
  fills from the slope at `phi(0)`. Of the twelve temperatures probed across the band, the
  seven that already had an action are unchanged to the last digit and the five that had
  none are back, on a smooth curve through them, with the dimensionless `dS_3/dT` falling
  monotonically from 1299.5 to 459.7. At T = 61.52 GeV the recovered 15705.829340759480
  agrees to 2.5e-06 with the 15705.790376807605 an analytic `SplinePath.dV` gives, the
  agreement the two also show at T = 70 GeV where both already worked. The orientation of
  the minima no longer decides whether a bounce exists: on the quartic
  `0.25 phi^4 - (1+d)/3 phi^3 + d/2 phi^2` at d = 0.1, with the true vacuum at 1, the solve
  was refused outright and now gives 1.7132091393850302 against 1.7132017403828275 for the
  same potential mirrored through `phi -> 1 - phi`. Both instanton examples in
  `tunneling1D`'s own module docstring take that orientation,
  `SingleFieldInstanton(1.0, 0.0, V, dV)`, and both raised `TooSmalldVError` on the first
  trial; they now return a profile.

- **The percolation estimate takes the hottest root of its criterion.**
  `calcApproxPercolation` places the first grid of the adaptive solver, and it handed the
  whole phase-overlap interval to `optimize.brentq`, which assumes one root. That interval
  carries several: the criterion is positive below the percolation point and negative above
  it; as `T` falls the rate vanishes, so
  `crit = Gamma^(1/4) / (H betaH (I_perc / 8 pi v_w^3)^(1/4))` tends to zero and `crit - 1`
  turns negative again at the cold end; and `T <= Tmin` returns `+inf` by construction.
  Nothing exposed that while the shooting aborted on a wrong-sign slope, because
  `if np.isinf(SdT) or np.isinf(S): return np.inf` fired inside the band where the bounce
  would not solve and manufactured a sign change in roughly the right place. Measured at
  `lambda3` = 5.815656566 over the interval [2.35254e-13, 121.839] GeV, the estimate
  returned 61.5057 GeV before the shooting fix and 1.96582e-12 GeV after it, thirteen and a
  half orders of magnitude too cold, and the first grid of the run that followed it was
  placed there. The three trees cost 26, 94 and 48 cached action temperatures at that
  coupling, and 41 of the 49 criterion calls of the middle one are below 1 GeV.

  The hottest root is now bracketed by walking down from the top of the interval in steps of
  `_APPROX_PERCOLATION_SCAN_RATIO` = 1.15 until the criterion changes sign, and only that
  sub-bracket is handed to `brentq`. Percolation is the first temperature the universe meets
  on the way down, so the hottest root is the one wanted and the others are artefacts of the
  bracket. The criterion's value at the top of the interval is `-inf` by construction rather
  than by measurement, since it answers "too hot" there before computing an action, so the
  first step brackets against that as a definite sign; read as no sign at all, a root within
  a factor 1.15 of the top is stepped over and a colder one returned in its place. The walk
  stops where `S_3/T` passes `_NUCLEATION_DEAD_S_OVER_T`, the threshold the controller's
  grid already uses, applied only once the walk has entered the band where nucleation is
  possible, because `S_3/T` is large at both ends of the interval for opposite reasons. It
  stops as well after `_APPROX_PERCOLATION_MAX_BLIND_STEPS` = 20 consecutive steps whose
  action could not be computed, which the first stop cannot see because it has no `S_3/T` to
  read. Only a probe that carries a sign becomes the bracket's hot end: an action that could
  not be computed answers `+inf`, whose bracket sign is zero, and letting it take that place
  makes every later comparison a product with zero, so a finite negative sample, a failed
  solve and a finite positive sample bracket nothing and the root between the first and the
  third is stepped over. The blind step is still counted, so a run of failures still ends
  the walk. Measured: with no computable action anywhere, crossing that interval costs 243 steps and 484
  action evaluations. Where there is no sign change at all the estimate raises
  `PercolationApproximation1Error`, which `_initial_temperature_grid` already answers by
  continuing from the hottest `TSYM` that brackets `Tb_criterion`. At
  `lambda3` = 5.815656566 that is the right answer: the criterion is negative at all 23
  temperatures between 92.1277 and 4.25635 GeV where the walk gets a finite value out of
  it, the largest being -0.998905, so the 61.5057 GeV was an artefact as well. Where the
  walk already worked the estimate is unchanged: at `lambda3` = 5.5 it returns 89.5357 GeV
  from 14 criterion calls and 22 cached action
  temperatures. `bubbledynamics_fixedstep.calcApproxPercolation` keeps the whole-interval
  call, as the fixed-step solver is brought onto this implementation in a change of its own.

- **The shooting bisection converges to a fraction of the `x` it brackets, not to an absolute
  window.** `findProfile` varies `x`, where the trial profile starts at
  `phi(0) = phi_absMin + exp(-x) delta_phi`, so for a small excursion `x` is the fraction of the
  gap between the minima that the bounce covers,
  `|phi(0) - phi_metaMin| / |delta_phi| = |exp(-x) - 1| ~ x`. On a supercooled transition that
  fraction shrinks with the temperature while an absolute `xtol` does not, so the loop delivered
  `x` to a relative precision of `xtol / x`: on the shipped conformal dark U(1) at g = 0.46,
  y = 0.01, v_GeV = 0.14 that runs from 1.6e-07 at T_int = 0.1 to 2.3e-02 at 1e-6. What that
  costs is measured in the number of bisection trials the absolute rule allows, counted as
  calls to `integrateProfile` on that model at the configured `xtol = 1e-10`, with only the
  stopping line put back to `(xmax - xmin) < xtol`: 25, 22, 19, 15, 12 and 8 at
  T_int = 1e-1, 1e-2, 1e-3, 1e-4, 1e-5 and 1e-6. The count falls as the transition cools,
  because once `x` is small an absolute window of `xtol` is already wide compared with the
  bracket, so the loop stops before the bisection has converged. With the window a fraction of
  `x` the count is 36 at every one of those temperatures. The window is now
  `xtol * min(max(|x|, |xmin|), 1)`, with `xmin` present so the test is defined before `x` has
  been bracketed from below; `xmax` starts at infinity, so the test cannot fire at all until an
  overshoot has bracketed `x` from above.

  The scale is capped at one so the window is never wider than the absolute `xtol` it replaces.
  Without that cap the rule is the looser of the two wherever the bracket sits above `x` = 1,
  which is the thick-walled end and not a corner case: on the same conformal dark U(1) at
  g = 0.7, y = 0.24, v_GeV = 1.0 the accepted trial passes `x` = 1 at T = 97.5 and reaches 2.02
  at T = 135, and without the cap the last two trials came out 1.0114 to 1.1598 times `xtol`
  apart at the five temperatures from T = 125 to 135. The cap costs one further trial there,
  37 against 36, and returns that gap to 0.51 to 0.58 of `xtol`. With it the window is
  `xtol * x` where the bounce is small and `xtol` where it is not, so this tightens the
  bisection on a supercooled transition, leaves it where the absolute rule already had it on a
  thick-walled one, and loosens it nowhere.

  The loop also stops when the bracket reaches the resolution of the floating-point grid, on
  `x == xmin or x == xmax`. A relative threshold needs it: `xtol * x` and the spacing `ulp(x)`
  both carry the factor `x`, so the threshold falls under the spacing as soon as `xtol` goes
  below 2.2e-16 whatever `x` is, where the absolute window only became unsatisfiable below
  `ulp(x)` itself, which on the supercooled line is 5.2e-26. Past that point the midpoint lands
  on a bracket endpoint and every further trial repeats the same `x`. Measured at
  `xtol = 1e-16` on the shipped conformal dark U(1) at g = 0.46, y = 0.01 and at g = 0.7,
  y = 0.24: without this test both pass 400 trials with the last six `x` identical, with it
  both stop after 55, and at `xtol = 1e-20` also after 55, since what ends the loop there is
  the spacing rather than the tolerance. At the configured `xtol = 1e-10` and at the signature
  default of 1e-4 the trial counts are unchanged, 36 and 16.

  Measured against a bounce solved a second way, the reflected construction described in the
  entry below, S3/T at g = 0.46 goes from +0.005, +0.063, +0.722 and +11.4 per cent at
  T_int = 1e-4, 1e-5, 1e-6 and 1e-7 to +0.000, +0.000, +0.004 and +2.0. At the two couplings
  that decide where the model percolates, at the temperature where they percolate, it
  reproduces the converged action to all six figures printed: 129.656 at g = 0.43 and 120.819
  at 0.44, against 129.761 and 120.921 without it.

  No verdict changes. The 22-coupling ladder is identical point for point, 17 of 22 with the
  lowest at g = 0.44, and `T_perc` moves by a median of 0.0001 per cent over the 17 that
  percolate, by 2.76 per cent at g = 0.44. On the 2HDM benchmark the cap makes this a no-op. At
  the three warm-edge couplings of the `edge` grid, `lambda3` = 5.77, 5.7713649314 and
  5.7727298627, the bracket converges at `x` = 2.27 to 2.45, above the cap, and every field of
  the recorded result reproduces the previous entry's code exactly; the one leaf that differs at
  each coupling is the wall time the run took. Without the cap those three differ in 12 of the
  15 summary fields, by up to 0.081 per cent in `(beta/H)_RH`.

  `tests/test_1d_tunneling.py::test_action_calculation` re-pins from 2388.030513420128 to
  2388.601280038287. That call takes the signature defaults, and at `xtol = 1e-4` this change
  moves the answer most, because at that tolerance neither rule has converged: `x` at the
  accepted trial is 0.0857, so the window is a factor 12 the tighter and the bisection takes 16
  trials against 13. Looser is not closer: at the model's own 10000 spline samples the distances
  from a converged 2384.3235 are 0.4248 before the previous entry's change, 0.0390 after it, and
  0.6498 here. At the configured `xtol = 1e-10` this code is the closest of the three,
  2388.1790354 against 2388.1789449 and 2388.1788707, which is what the test pins.

- **The bounce profile is integrated, and judged to have arrived, on the scale of the bounce
  and not of the gap between the minima.** `findProfile` holds the profile integration to an
  absolute error `epsabs`, and that was `phitol` times `delta_phi`, the distance between the two
  minima. The field range the integration covers is the bounce's own excursion,
  `|phi(0) - phi_metaMin|`, which on a supercooled transition is a few `T` while the minima stay
  `v` apart. On the classically conformal dark U(1) at g = 0.46, y = 0.01, v = 0.14 GeV -- the
  configuration every number in this entry is measured at -- the gap is 1000 internal units at
  every temperature while the excursion is 4.3 to 6.1 T, so `phitol |delta_phi|` is 1.6e-07 of
  the integrated feature at T_int = 0.1 and 2.4e-02 of it at 1e-6. `epsabs` is now computed
  inside the shooting loop from the current trial bounce.

  `integrateProfile` spends `epsabs` on two different jobs, and both were mis-scaled. Besides
  controlling the Runge-Kutta step it is the radius in `abs(y - [phi_metaMin, 0]) < 3*epsabs`
  inside which a trial profile counts as having reached the false vacuum, and that verdict ends
  the shooting loop. Tied to the gap, that radius was a fixed absolute distance while the bounce
  shrank with the temperature, so `3 phitol |delta_phi| / excursion` grew to above one by
  T_int = 1e-7: an acceptance radius wider than the whole bounce. The first trial was then
  declared converged with the profile having barely moved.

  Changing one without the other does not work, and makes things worse at some temperature.
  Giving the bounce's scale to the Runge-Kutta step alone, with the acceptance radius left on
  the gap, takes S3/T at T_int = 1e-6 from 96.45 to 91.10 -- further from the converged 102.46,
  because a better-integrated profile still stops at the same too-wide radius -- and still
  returns zero at 1e-7. Giving it to the acceptance radius alone returns 96.93 at 1e-6 and
  636.3 at 1e-7. Only the two together give 103.20 and 102.93. Tied to the bounce,
  `3*epsabs/excursion` is `3*phitol` at every temperature.

  One consequence is worth stating: that acceptance test now never fires. It asks for
  `3*phitol` of the excursion in `phi` *and* in `dphi`, which a shooting method that stops on
  its own bracket does not reach, so the loop ends on `(xmax - xmin) < xtol` at every
  temperature measured. Replacing the test with `False` leaves the conformal action identical to
  the last digit at T_int = 0.1, 1e-4, 1e-6 and 1e-7 and leaves the whole test suite unchanged.
  It was only ever reached because it was scaled to the gap, and when it was reached it was
  wrong.

  How far this reaches a transition that is not supercooled depends on how thick its wall is,
  and it is not nothing. On the g = 0.7, y = 0.24 configuration at T = 15 that
  `tests/test_1d_tunneling.py` uses, where `T/v` is 0.015 and the wall is thick, the two rules
  differ by 3.1e-08 relative at the configured `phitol = 1e-10`. On the 2HDM benchmark's
  supercooled grids, 97 couplings of the `edge` and `refined` lines at `T/v` of 0.082 to 0.182,
  they differ by more. Over the 72 pairs that complete on both arms the percolation temperature
  moves by a median of 0.0039 per cent, with a 90th percentile of 0.40 and a largest move of
  34.5; `(beta/H)_RH` by a median of 0.17 per cent with a largest of 99.8; `alpha` by a median
  of 0.016 per cent with a largest of 462. The median is the no-op this change was expected to
  be and the tail is not, and the tail is at the cold end of the two grids that were measured,
  `edge` and `refined`, where `T_perc` falls from 37.136 GeV to 20.0 GeV over `lambda3` of
  0.0061 and the spectrum is a steep function of it. The benchmark's `marginal` grid and its two
  `broad` grids are not in this campaign.

  8 of the 97 points change verdict: `edge 5.7918389018`, and on `refined` 5.8099242424,
  5.812995338, 5.8157252007, 5.8167488993, 5.819478762, 5.8225498575 and 5.8314219116. Four are
  points `main` resolves and this code does not, three the reverse, and the last fails on both
  with only the code changing from 10 to 8. Which arm is right at each one is not established
  here: seven of the eight sit at the cold end, where each arm's own scatter against a local fit
  through its neighbours has a 90th percentile of 3 to 4 per cent in `T_perc`, against 0.003 per
  cent warm of 37.5 GeV, and the sign of that residual changes with the choice of fit at five of
  the six points where the test can be made.

  The largest move does not rest on the neighbours. At `refined lambda3 = 5.8184550635`, `main`
  returns `T_perc = 41.012` GeV and `(beta/H)_RH = 2058.3` against 26.845 GeV and 4.5467 here.
  `main` placed 53 of its 197 support points inside 41.00 to 41.10 GeV, a window of
  `dT/T = 2.16e-03`, and across it its own `S3/T` runs from 106.39 to 151.40; its recorded
  `(beta/H)_S3` of -67.4 allows 0.15 units over a window that narrow, and 45.0 units is `e^45`
  in the nucleation rate. This code has one support point there, at `S3/T = 150.04`. Any
  statement that quotes a single `edge` or `refined` point of this benchmark has to be re-read
  against this branch rather than against `main`.

  Measured against a bounce solved a second way -- the false vacuum at the origin,
  `V(phi) - V(0)` of the unsplined potential, a wall at 30 times the escape point, and the
  problem reflected so that the minima handed to the solver are the bounce's own scale rather
  than `v`, which is a construction that cannot carry this defect -- S3/T at g = 0.46, y = 0.01
  was low by 0.003, 0.021, 0.144, 0.794 and 5.865 per cent at T_int = 1e-2, 1e-3, 1e-4, 1e-5 and
  1e-6. It is now +0.000, +0.000, +0.005, +0.063 and +0.722 per cent at the same five
  temperatures, and the collapse at 1e-7 is gone. At T_int = 1e-1 both rules sit 0.55 per cent
  above that reference, the same offset on each, which is the path spline against the unsplined
  potential and not these tolerances.

  What is left of the 0.722 per cent at T_int = 1e-6 is not `epsabs`. With the loop no longer
  ending on `3*epsabs` what stops it is the shooting bracket `(xmax - xmin) < xtol`, which is
  absolute in `x` while `x` itself is 4.3e-09 there, so `xtol = 1e-10` resolves the escape
  point to 2.3 per cent only. Judging that bracket relative to the `x` it contains, at the same
  `xtol`, gives 102.465 against the reference's 102.461: +0.004 per cent in place of +0.722,
  and the same verdict at every coupling of the ladder. That is a separate change and is not
  made here.

  **This reduces the couplings at which the conformal dark U(1) percolates, from a lowest
  g = 0.42 to 0.44**, and that is the correction rather than a regression: at the temperature
  those couplings percolate, internal 1.8e-6, the action was 7.45 per cent low at g = 0.44,
  S3/T 111.823 against the reflected reference's 120.819, and 7.90 per cent low at g = 0.43,
  119.415 against 129.656. Since `Gamma ~ exp(-S3/T)` at `S3/T` of 120 to 130, that
  understatement raised the nucleation rate by 8.1e+03 and 2.8e+04 respectively. This code
  reproduces that reference to 0.08 per cent at both couplings, 120.921 and 129.761. `Tperc`
  moves by -5e-05 per cent at g = 0.62 and -47 per cent at g = 0.44, the coupling now at the
  boundary.

  It takes longer, because tighter tolerances mean smaller integration steps on the same solves.
  Measured on `examples/example_point.yaml`, the conformal dark U(1) at g = 0.7, v_GeV = 0.1,
  y = 0.01: 143.04 s at d66a774 against 183.92 s here, 28.6 per cent, on one core of an idle
  machine for observables identical to five figures. That is the same point run twice and is
  the only runtime figure quoted here. A per-grid cost over the 2HDM benchmark is not quoted
  because it is not measurable: the two arms' runtimes per coupling scatter by 21 to 38 per
  cent for reasons unrelated to this change, and over 592 pairs on four grids no grid total is
  distinguishable from zero.

  The extra time is inside the integration rather than around it: counted through `run_TL`,
  this code makes 1036 `calcAction` calls against 1059 and 3117 `integrateProfile` calls
  against 3150. The subprocess timeout in `tests/test_release_smoke.py` is raised from 180 s
  to 600 s accordingly, with all six of that test's reference bands unchanged.

  Not addressed: below about T_int = 1e-8 the action is wrong by orders of magnitude on both
  sides of this change. At g = 0.44 and T_int = 1e-8 both return S3/T = 5.6e+08 against 96.50
  from the reflected reference above, and the solver has lost the bounce rather than
  mis-integrated it. The barrier there ends 1.36 T from the false vacuum, measured from
  V(phi) - V(0) alone with no solver and no tolerance in it, against 1.49 T at T_int = 1e-6
  where the solve works. The profile the solver returns starts 1.4e+07 T out, against 4.3 T at
  1e-6 and 4.8 T at 1e-7: it has gone four orders of magnitude past the whole barrier into the
  region where the potential falls away, and the action it reports belongs to that excursion
  and not to a bounce. Nothing below T_int ~ 1e-7 should be read as converged, and the
  percolation of a model whose transition completes that cold is not established here.

- **A timeout raised inside the initial-condition root find is no longer read as a wall too
  thin to integrate.** `errors.Timeout` comes from a POSIX alarm handler, so it lands wherever
  the run happens to be, `scipy.optimize.brentq` included, and it subclasses `Exception`. The
  handler around that root find in `SingleFieldInstanton.initialConditions` turned it into a
  `TunnelingError`, which `pathDeformation._find_profile` turns into an `InfiniteActionError`
  and `bounceAction` reports as no tunnelling: a timed-out point came back as a finished one
  with a verdict. It is re-raised, as it already was in the path extension and the spline fit.

- **Two temperatures are compared relative to their own size, not to an absolute floor of one
  and not to the hottest point of the support.** The adaptive percolation decides whether a
  proposed temperature is already present in the support, whether two samples are duplicates,
  and which support point an interval endpoint refers to. Every one of those comparisons used
  an absolute tolerance built either as `1e-6 * max(..., 1.0)` or as `1e-6 * max|support|`.
  Temperatures are in the model's internal units, where the stable minimum sits at `v_stable`
  and a `conversionFactor` converts to GeV, so an internal temperature is `v_stable` times
  `T/v` and falls below one wherever the transition is supercooled by more than `1/v_stable`:
  4e-3 for the 2HDM, 1e-3 for the models built on `internal_scale = 1000`. Both tolerance forms
  then become a large fraction of the temperature they are applied to.

  The floor of one made the presence tolerance 1e-6 in absolute terms, larger than the
  temperature itself below internal 1e-6. The support maximum is worse, and worse on models
  that are not extreme: it only needs the support to span decades, which it does whenever the
  bank reaches up to the critical temperature while the transition happens far below it. The
  2HDM reaches both regimes. On the 144-point support of the point at `lambda3 = 5.8197`,
  which reached internal 2.4e-13 while its hot end stayed at 76.2 -- the 2HDM's
  `conversionFactor` is one, so those are GeV too -- 32 points sat below internal one, the
  presence tolerance was 7.6e-5, and 26 of the 143 adjacent pairs of the support's own points
  fell inside it: 22 of them more than a tenth apart, the median one 172 per cent apart, and
  the three coldest 110 per cent apart each. The rule that replaces it merges four of the same
  143, all duplicates to nine digits. A 200-point log-spaced support from internal 1e-12 to
  1e-1 has 125 of its 199 adjacent pairs inside the old tolerance and none inside the new one.
  Either way, every point the jump refinement proposed inside a rise of `P(T)`
  counted as already present and was dropped, the controller found nothing to add with most of
  its budget unspent, and the acceptance gate then rejected the profile the controller had
  accepted.

  Each comparison is now `|a - b| <= relative * max(|a|, |b|)`, through one helper
  `_temperatures_match`, with `relative` 1e-6 for presence and 1e-10 for deduplication. The new
  tolerance is never larger than either of the old ones, so points can only be separated, never
  merged: the change can add support points and action evaluations, never remove them. It is
  bit-identical wherever every temperature involved is at least one, which is where the floor
  never acted; it is not identical on a support that spans decades, whatever its absolute
  scale, which is the case it exists for. The 2HDM points that percolate are on the safe side
  of the old boundary and do not move: on the support of the point at `lambda3 = 5.8106060606`,
  which spans internal 2.2 to 76.2, the old rule merges three adjacent pairs and the new rule
  merges the same three, and the campaign reproduces that point's observables unchanged. It is
  the points that fail to percolate whose supports reach far below one, and there the two rules
  part: 25, 28 and 26 of their adjacent pairs merged under the old rule against 5, 5 and 4
  under the new one. Across the four dark U(1) regression datasets of the validation scans,
  internal `Tperc` runs from 0.21 to 162 against an internal `Tcrit` of 84 to 316, so the floor
  of one is nearly inert there -- 22 of their 1080 points fall below internal one -- while the
  old presence tolerance still reached 5e-5 to 1.5e-3 of the cold end; a refinement step
  coarser than that is unaffected, and the jump refinement's steps are of order a per cent.

  Two further absolute floors are removed for the same reason, and both are latent rather than
  active: the dead band around the step-1 startup centre, which at an internal window of 1e-9
  stops `_count_high` and `_count_low` from seeing points that are there and makes the fill
  loops spend action budget replacing them, and the window that collects cached actions, which
  at an internal window of 1e-9 reaches a tenth of the way below it and admits actions from
  outside it. Both are 1e-10 absolute, so they need an internal window below about 1e-8 to
  matter, which is colder than any model shipped here reaches; they are removed because the
  expression is the same one and the next supercooled point is where it would be found. The
  lookup that turns
  the endpoints of a probability jump back into support indices took the first match of an
  absolute tolerance scaled by the hottest support point; on the 144-point support above it
  resolved 28 of the 143 adjacent pairs to at least one wrong index, 26 of them to
  `high_index >= low_index`, which returned "not inert" -- the verdict that vetoes the repair of
  the jump -- without reading a rate. It now matches relative to each endpoint and takes the
  closest support point.

  A fixed-step scan is untouched by all of this. `percolation_algorithm_mode="fixed_step_size"`
  selects `transitionObservables_fixedstep`, which calls `calcPercAndEvolve` from
  `bubbledynamics_fixedstep`, and that module contains none of the changed names: not
  `_temperature_grid`, not `_finalize_percolation_state`, not the presence rule, the window
  tolerances or the interval-endpoint lookup.

  Those last two do reach the other fixed-step solver. `bubbledynamics.calcPercAndEvolve` runs
  `percolation_fixedstepsize` when it is handed a fixed-step configuration, and there
  `_temperature_grid` builds the support bank and `_finalize_percolation_state` applies the
  1e-10 deduplication afterwards. That path is reached from the adaptive observables module, not
  from a fixed-step scan, and both functions can only keep temperatures the old rule merged, and
  only where a temperature below one in internal units is involved.

- **A phase no longer leaves the range of its own traced minima between two traced
  temperatures.** `PhaseInfo` interpolated the trace with `splprep(..., s=0, k=3)`, an
  interpolating cubic. The tracer does not space its temperatures evenly -- it clusters them
  where it steps small and leaves gaps where it steps large -- and a cubic through spacing
  ratios of that size oscillates between the nodes. The minima are now interpolated with
  `PchipInterpolator`, which is shape preserving: between two traced temperatures each field
  component stays between the two values traced there. It still passes through every traced
  node, so this is not smoothing, and past the ends of the trace it still extrapolates rather
  than refusing: two places in `bubbledynamics` read a raise from `valAt` as a phase that
  cannot be evaluated at that temperature, and clamp their stencils for that reason, so a
  refusal there would be taken for a missing phase.

  The bounce solver reads the phase through `valAt`, so an overshoot is handed to it as a field
  value the tracer never found: the solver then tunnels from a displaced point rather than from
  the one the tracer established, a shorter path over a lower barrier, and returns too small an
  action over the temperatures either side of the excursion. With the shape-preserving
  interpolation there is no excursion: the largest field value between two traced temperatures
  is the largest of the two. A trace that exhibits the excursion is the fixture of
  `tests/test_phase_interpolation_does_not_overshoot.py`, where the two interpolations are
  compared on it directly. That trace comes from a development model outside this repository
  and is kept as recorded data, not as something reproducible from the models shipped here.

  A displaced action is what produces the steps the rate-jitter gate flags, so removing the
  overshoot removes one cause of them. The comment at the action discard added in the
  percolation work named this overshoot as the reason couplings reached it and said the fix
  belonged in the phase tracing; it is updated to say the fix has landed, and the discard is
  kept for any other model that trips the gate.

  This changes results. Where the trace is evenly spaced the two interpolations agree closely;
  the difference grows where the tracer left a cluster next to a gap, which on a supercooled
  line is the stretch nearest the end of the transition. Per-model figures are not quoted here:
  the campaign behind them loaded its potential from a development checkout rather than from
  this repository, so its numbers describe a different model. A re-validation against the models
  shipped here is running, and measured figures will be added when it lands.

  Both solvers are affected: `phases.py` has no fixed-step twin, and
  `transitionObservables.py` and `transitionObservables_fixedstep.py` both read this one
  `PhaseInfo`. `PhaseInfo.tck` is replaced by `PhaseInfo.interpolator`; it was read nowhere
  outside `phases.py`, and of the 114 `valAt` call sites in `src` exactly one passes `deriv`,
  which the new interpolator supports. `valAt` keeps its signature and the shape it returns
  for a scalar, one- or two-dimensional `T`; its `deriv` argument now accepts any order rather
  than at most three, returning zero above the third. A trace carrying a non-finite
  temperature or field value is refused when the phase is built, where the B-spline accepted it
  and returned `nan` at every temperature of the affected component; scanning turns that into
  an error for the point rather than `nan` observables.

- **The bubble nucleation rate no longer grows above the temperatures whose action was
  computed.** The percolation result's action spline returned the action of the hottest
  temperature that had one for everything above it. Holding the action constant while the
  temperature rises lets `S3/T` fall, and the rate goes as `exp(-S3/T)`, so the rate grew
  towards high temperature where the barrier grows and it has to vanish. It is now infinite
  there, which is the limit and is what the estimator's own spline in
  `percolation_adaptivestepsize` already returned, so the two agree. The upper limit of the
  mean-bubble-separation integral is likewise the hottest temperature whose action is known
  rather than the hottest in the support, since a temperature whose solve failed carries no
  integrand. Only the adaptive backend is affected: the fixed step size backend builds its
  action spline over the finite-action temperatures with `fill_value = (-inf, +inf)`, so it
  already returned an infinite action above its data and its bubble-number integrand is
  identically zero there.

  The two only differ where an action solve fails at the hot end of the support, which was one
  coupling of the 333 resolved across the validation scans: the 2HDM benchmark at
  `lambda3 = 5.7727298627`, where the solve failed at T = 57.18 GeV while the hottest computed
  action was at 49.28 GeV. The action was held at 7441.88, `S3/T` fell from 151.0 to 130.2 over
  that stretch, a factor `exp(20.8)` in the rate, and the integrand of the bubble-number
  integral peaked at the hot end with 99% of its weight there instead of at the percolation
  temperature. The bubble number density came out 7340 times too large, which put the mean
  bubble separation 19 times too low at `RH = 0.002352` against 0.044533 and 0.046287 at the
  neighbouring couplings, `betaH_RH` at 1883 against 99.4 and 95.7, and the LISA signal-to-noise
  ratio at 7.8 against 23.9 and 26.5. With the fix that coupling gives `RH = 0.045396` and
  `betaH_RH = 97.53`, between its neighbours at 99.42 and 95.65, and the LISA
  signal-to-noise ratio 25.18 between 23.92 and 26.49. Where no solve fails the two rules
  agree and nothing moves: over the four U(1) scans behind figures 4, 12 and 13, 1082 points
  in all, `Tcrit`, `Tnuc`, `Tperc`, `Treh`, `alpha`, `betaH_S3`, `c_s` and `v_wall` are
  bit-identical at every point and every error code is unchanged, while `RH` and `betaH_RH`
  move at 51 of the 1080 points that resolve, by at most 4.7e-04.

- **The bounce action no longer dies where the barrier is small compared with the path.**
  `pathDeformation` set several of its numerical scales to a fixed fraction of the distance
  between the two minima, while the barrier is a feature whose width is set by the temperature,
  so in a strongly supercooled model each scale eventually exceeded it. Reported from the
  four-dimensional daisy-resummed rate of the conformal U(1), where the action came back
  infinite below T_int = 0.146 at g = 0.54 with the solver calling the potential "stable, not
  metastable" although the barrier was there. On the conformal dark U(1) ladder in `g`
  (`y = 0.01`, `v = 0.14 GeV`), against `main` at b31e178, the lowest coupling that percolates
  moves from 0.54 to 0.42: twelve couplings gain a percolation, 0.53 from "incomplete" and 0.52
  down to 0.42 from "no percolation". Two more change one failure for another, g = 0.38 and 0.36
  from "no percolation" to an unresolved jump in P(T), and g = 0.40 does not percolate on
  either side. Where the old code also resolved it the percolation
  temperature is unchanged, by 1.8e-06 to 2.4e-04 per cent over g = 0.54 to 0.62, the one
  exception being -0.074 per cent at g = 0.55.
  Against TL's own `SingleFieldInstanton` on the unsplined potential the action is now right to
  1.3e-08 at T_int = 0.1 and 0.01, 4.2e-07 at 1e-3, 6.6e-08 at 1e-4, 2.6e-07 at 1e-5 and
  5.9e-05 at 1e-6, where before it was infinite below T_int = 0.1. Those figures measure the
  path and the spline, not the whole solve: the reference carries the same shooting tolerances
  as the solver under test, `tunneling_findProfile_params` `phitol = xtol = 1e-10`, and those
  are themselves fractions of the distance between the minima. Against a bounce converged with
  tolerances relative to its own profile, the action is biased low and smoothly so below about
  T_int = 1e-2: at g = 0.46, 131.713 against 131.899 at T_int = 1e-4 and 96.76 against 102.854
  at 1e-6. That fourth scale is not touched here, so the numbers above should be read as the
  accuracy of the path construction at a fixed shooting tolerance, and the action below
  T_int = 1e-2 is not yet converged.

  - The path extension searches for the nearest minimum. It used `scipy.optimize.fmin` from
    `x0 = 0`, whose initial simplex step is 2.5e-4 of the knot spacing whatever the problem
    looks like. On a two-knot path that spacing is the whole path, so the end extension settled
    a full knot out, at the mirror of the broken minimum, because the potential is even in the
    field: the path ran from 1000 to -1000, connected two minima degenerate at -2.0220e+08, and
    the solver reported no metastable end. A rise now has to clear the rounding of the
    potential, since near an end it can be flat to machine precision over decades of the
    distance, and an extension past the metastable end is refused if it would take the potential
    to or below the other end, which is what destroys the metastability the bounce needs. An end
    with no minimum within reach is left alone.

  - The samples that build the potential spline are placed where the barrier is, but only where
    the uniform grid is blind to it: fewer than two of them inside the barrier and at least a
    thousand of them in total. Both bounds are measured. The overhang past each end shrinks with
    them, and is the larger effect of the two: the potential is -2.0e+08 at the broken minimum
    and -9.3e-17 at the symmetric one, and 0.2 of the path past the symmetric end it is
    -2.4e+06, so one cubic spline through all of it is bent, just inside that end, towards a
    value far above the barrier. A path whose barrier the uniform grid already carries keeps
    exactly the samples it had, which is what leaves the 2HDM unchanged, to 2.0e-06 at T = 85.3
    and 8.7e-06 at T = 48.8.

- **The adaptive percolation controller no longer spends its budget below the transition,
  and no longer refuses to repair a jump it then rejects the profile for.** Eight separate
  faults, all found by instrumenting a 40-point 2HDM "edge line" scan on which 31 of 40
  points resolved.

  - A percolation sample whose `H * T * 3 c_s^2` has underflowed to exactly zero is marked
    invalid. Every factor can be positive and finite while the product underflows, and
    `-log(0)` is `+inf`, which the floor clamp's `np.maximum` removes for `-inf` but not for
    `+inf`; it then reached the PCHIP interpolants as a bare
    `ValueError: 'y' must contain only finite values` with no typed error and no diagnostic.
    The bag-limit branch of the same integral does not share the exposure: on the grid that
    makes one of 32 general-EOS denominators underflow, its `log(1/H)` transport term is
    still finite at 667.7 and its source term never goes `+inf`, because `-S/T` dominates.
  - `_refine_step1_rate_peak` drops non-finite samples before building its interpolant. Its
    peak search is `np.nanargmax`, so it is written for an array that may carry them; ten
    lines later it handed the same array to `PchipInterpolator`. Captured at
    lambda3 = 5.8191375291 with 1 NaN among 79 samples.
  - The step-1 scout hands step 2 the temperatures whose action came back *usable* as free
    support, not every temperature it *probed*. The free bank is support that costs no new
    action budget, so a failed cold probe handed over there plants the active grid's cold
    boundary on it: traced from `[2.2235, 62.279]` to `[3.8032e-06, 68.772]` GeV against a
    transition at 20-40 GeV, after which the rate-step refinement filled the span in one
    step (29 to 75 points).
  - A support point whose solve has already run no longer counts against the controller's
    new-action budget when that solve returned non-finite. Its cost is sunk, and counting it
    as pending held the whole budget for the rest of the run.
  - The cold range expansion stops where the bounce stops existing. Below the transition the
    symmetric phase is no longer metastable and every action there is non-finite; those
    samples are dropped by the percolation integral and by the profile rebuild, yet each one
    costs a bounce solve. 38-57% of all support points in every one of five recorded runs
    were of that kind, in one contiguous run at the cold end.
  - A cached action is no longer promoted into the active grid where `S_3/T` has run past
    anything that can nucleate. The cold actions of a completed transition do not fail, they
    saturate: measured at lambda3 = 5.8123128723 the action settles near 4357 and `S_3/T`
    reaches 1.15e+09 at T = 3.8e-06 internal, so the point has a perfectly finite action and
    a rate of exactly zero. The test is confined to candidates colder than the grid already
    is, because `S_3/T` is just as large on the hot shoulder: across the 31 recorded
    edge-line profiles the hottest retained support point sits between 478 and 509.
  - The rate-step refinement no longer refines an interval whose colder end has collapsed
    more than 1000 decades below the profile's own rate peak. That floor is sized for
    deciding where to *place* support, where being permissive costs only a few wasted
    evaluations; the rate-jitter gate judges smoothness instead and carries its own, much
    tighter floor of 30 decades, under *Changed* above. A fall out of the live region exceeds any
    `max_log10_rate_step`, so the refinement walked lambda3 = 5.8205024605 down to
    T = 1.4e-5 GeV with `log10(Gamma/H^4) = -1.4e8`, whereupon one quadratic in `ln T`
    fitted across seven decades of temperature reported a residual of 1.6e10 orders of
    magnitude and advised raising the bounce precision for a sample at fourteen
    microelectronvolts. The floor is relative to each profile's own peak, never absolute:
    contributing samples reach `log10(Gamma/H^4) = -447` in absolute terms, and a strongly
    supercooled model has its peak at very low temperature by construction.
  - A `Delta P` jump that the acceptance test will reject is now repairable even past the
    rate peak, where the post-peak inert veto used to decline it unconditionally. The two
    rules contradicted each other: at lambda3 = 5.8177725978 the interval
    [28.745, 30.453] GeV lies below the rate peak at 41.05 GeV, so the controller drove
    max `Delta P` from 0.85 to 0.12, found nothing further it was permitted to do, and
    accepted the profile with 119 action evaluations unspent, whereupon the gate rejected
    it. "Post-peak" cannot excuse a coarse `P(T)` in any case, since `R_H` is sensitive to
    `P(T)` above `Tperc` and about 97% of that interval lies inside its integration range.
    Where a repair exists only by this exemption the anchor remesh is offered first and the
    repair stays as the fallback, because at lambda3 = 5.8136778037 the repair otherwise
    took the slot the remesh needed and the run ended one support update short at
    `Delta P = 0.0578`.

- **A rejected profile says which resource ran out.** The unresolved-jump rejection asserted
  that the controller "already spent its remaining support budget on local jump refinement",
  which is frequently untrue and sends the reader to the wrong place: at
  lambda3 = 5.8136778037 the point was rejected with 115 of its action evaluations still
  unspent, having instead reached the support-update limit (`maxit`, then 10 and now 20).
  The message now
  reports whichever resource actually ran out, and says nothing about a cause when the caller
  does not supply the numbers. Reaching the support-update limit with a repair still waiting
  is reported as well, naming the repairs that were outstanding.

- **The daisy term is evaluated without throwing away its significant digits, and the sound
  speed is taken from the thermal part of the potential.** Both are numerics: the same
  expressions, evaluated so that they keep their accuracy. Three cancellations stood on top of
  one another.

  The Arnold-Espinosa term is proportional to `(m_T^2)^{3/2} - (m_0^2)^{3/2}`. Written that
  way it subtracts two nearly equal numbers wherever the thermal correction is small against
  the mass, which is the whole low-temperature regime: on the conformal dark U(1) of
  `examples/example_point.yaml` at an internal temperature of 0.02 the two cubes are about
  `1e9` while their difference is about `0.3`. It is now evaluated as
  `(m_0^2)^{3/2} [(1 + u)^{3/2} - 1]` with `u = Pi/m_0^2`, through `expm1(1.5 log1p(u))`.
  That rearrangement is kept in real arithmetic, because numpy's complex `log1p` and `expm1`
  lose a relative `8e-8` at `u = 1e-9` against `1e-16` for the real ones, and the `+0j` that
  exists only to admit tachyonic modes was being applied to every mode.

  The thermal correction itself was taken as the difference of the mass spectra at `T` and at
  zero, which keeps seven of sixteen digits at that temperature and **underflows to exactly
  zero** below an internal temperature of about `1e-7`, silently deleting the daisy term.
  `generic_potential.debye_massSq(X, T)` provides it directly. A model that writes its Debye
  masses in closed form overrides it, as `models/TL_conformal_dark_u1.py` does, and the base
  class measures it for every model that does not: where the thermal part is `Pi = c(X) T^2`,
  which is what the Arnold-Espinosa Debye masses are, the coefficient is read off at the
  model's own scale, where the subtraction still keeps its digits, and then used at any
  temperature.

  Nothing here assumes the quadratic law. It is verified, at a second temperature and at two
  field points, and the base class declines whenever the check fails. The check decides four
  of the eight models in `models/` each way. It passes for the conformal dark U(1), both dark
  U(1) parameterisations and the template. It fails for the three 2HDM models and for the dark
  flip-flop, and for one reason in all four: their bosonic masses are eigenvalues of a mass
  matrix whose *entries* go as `T^2`, and the eigenvalues of such a matrix do not. On
  `models/TL_2HDM.py` the two longitudinal gauge modes move their apparent coefficient by 4.6%
  and 11% between `T` and `T/2`. `models/TL_dark_flipflop.py` is the reason the check uses two
  field points rather than one: along `X0` its second field vanishes, its matrix is diagonal
  and the law holds to `7e-18`, while at a point with both fields on it is off by `5e-3`, so a
  one-point check would have accepted it and handed the daisy term a `Pi` wrong by half a per
  cent. For those four there is no `Pi` to hand over and the subtraction is the only route, so
  declining is the correct answer rather than a missing feature.

  The quadratic law is checked mode by mode, with a relative tolerance and a round-off-sized
  absolute floor for the modes whose coefficient is exactly zero. A single tolerance scaled to
  the largest coefficient only ever tests that one, and a mode whose coefficient is orders of
  magnitude smaller could be badly non-quadratic and still pass, after which a heavy mode can
  make that wrong contribution matter at low temperature. None of the eight models in
  `models/` distinguishes the two rules, the worst per-mode deviation among those that pass
  being `8e-15`, so this guards a case none of them exhibits.

  The verdict is settled once per model and at field points the model itself fixes, not at
  whichever point arrives first: `Vtot` is called with random field values while a model is
  being constructed, and a verdict read off those would differ between runs of the same input.
  Models whose coefficient is field independent, which is all four that pass, cache it and
  never measure again. Measured on the conformal dark U(1), the base class reproduces that
  model's own closed form to `1e-12`, and on the others it agrees with the subtraction to
  `1e-13` wherever the subtraction still works.

  This changes the daisy term of `models/TL_dark_U1.py`,
  `models/TL_dark_U1_g_parameterization.py` and `models/templatePotential.py`, which now get
  an exact thermal part where they previously got the difference of two spectra. A scalar
  `Vtot` costs 9 to 13% more on those, and 1% or less on the models that decline, since the
  verdict is cached. Where a model declines, the thermal part is still taken as the difference
  of the two spectra, as before, and the rearrangement above still improves the accuracy of
  that difference once it is taken.

  On the conformal dark U(1) an exact `Pi` changes the thermal part of the potential by
  `1.7e-3` of itself at a temperature of `1e-7` of the scale, and at `1e-9` the version
  without it has nothing left to compare. It changes `Vtot` by nothing measurable at any of
  them, which is the point: the shift lives below the last digit of the vacuum energy, and
  that is why the sound speed is taken from the thermal part. And
  `models/TL_conformal_dark_u1.py` implements it with the same `T^2` coefficients its mass
  function already uses.

  And `Hydrodynamics.calc_cs` now differentiates `generic_potential.V_thermal`, the
  temperature-dependent part of the potential, rather than the whole of it. A model that
  rewrites the effective potential from scratch is told by this class to override `Vtot` and
  `V1T_from_X`. Such a model must now override `V_thermal` as well, which is the
  temperature-dependent part on its own and is what the sound speed, the pseudo-trace
  strengths and the wall velocity are taken from; a reconstruction from the mass spectrum
  would not be that model's own thermodynamics. It is a separate hook from `V1T_from_X` on
  purpose: the class documentation calls the latter the temperature-dependent part of `Vtot`,
  which would include the radiation bath, while its base implementation returns neither the
  bath nor the daisy term, and nothing in the signature says which an override means. Adding
  the bath to an override that has it double counts, and not adding it to one that does not
  drops it, so a model that has rewritten `Vtot` or `V1T_from_X` without also overriding
  `V_thermal` is told so with an `errors.PotentialError` rather than handed a guess. No model
  in `models/` overrides either, so none is affected. The two are the
  same analytically, since what does not depend on temperature drops out of both derivatives,
  and differ numerically: the whole potential subtracts the vacuum energy from itself, and
  under supercooling that vacuum energy is the larger by ten orders of magnitude or more.

  And the derivative step is sized for what is being differenced. `temperatureDerivativeStep`
  sizes it from the magnitude of the whole potential, vacuum offset included, with a
  sixth-root balance derived for the five-point stencils of `dVdT` and `d2VdT2`; below a
  temperature of `1e-3` of the scale it saturates its own `3e-2` ceiling.
  `helper_functions.thermalDerivativeStep` replaces it for the thermal part. There the
  binding balance is the round-off of the second difference against the truncation of the
  first, `4 eps |V_th| / dT^2 |d2V_th|` against `dT^2 |d3V_th| / 6 |dV_th|`; the fourth
  derivative that would set the second difference's own truncation nearly vanishes wherever
  the daisy term dominates, because there `V_th` goes as `T^3` and `n(n-1)(n-2)(n-3) = 0` at
  `n = 3`. For `V_th` proportional to `T^n` that balance gives
  `dT/T = (24 eps / n(n-1)^2(n-2))^{1/4}`, in which the magnitude of `V_th` cancels because
  round-off and truncation both scale with it. The step is therefore a pure number:
  `(eps/3)^{1/4} = 9.3e-5` for radiation, `(2 eps)^{1/4} = 1.45e-4` for a daisy-dominated
  `T^3`, and the radiation value is used for both.

  Measured together, on the second temperature derivative of the daisy term, as the relative
  change when the derivative step is changed by a factor ten, released against fixed, on the
  traced broken phase of the conformal dark U(1) at `g = 0.7`:

  | `T/v` | as released | fixed |
  | --- | --- | --- |
  | `1e-1` | `3.7e-6` | `1.1e-7` |
  | `1e-2` | `7.7e-7` | `1.9e-9` |
  | `1e-3` | `7.5e-7` | `6.6e-9` |
  | `1e-5` | `5.6e-3` | `5.8e-9` |
  | `1e-7` | `1.1` | `2.6e-9` |
  | `1e-9` | the daisy term has underflowed to zero | `1.5e-9` |

  And on the sound speed itself: at this step `c_s` agrees to `4e-8` or better, on the traced
  broken phase, with the
  plateau measured over `dT/T` from `3e-5` to `1e-3`, at every temperature from `1e-1` to
  `1e-9` of the scale and at `g = 0.455`, `0.7` and `0.95`. Round-off takes over below
  `dT/T ~ 1e-5` and truncation above `3e-3`.

  `bubbledynamics.calcSoundSpeedSq`, which both percolation backends use for the sound speed
  that normalises the pseudo-trace strengths, differentiates the thermal part as well. Before,
  the adaptive backend took the whole potential at the step rule above and the fixed-step one
  took it at a hard-coded `dT/T = 1e-3`; the two now agree by construction. Where the released
  version returned nothing there, `alpha_thetabar`, `alpha_hyd` and `alpha_hyd_wall` were
  not-a-number and are now finite.

  In the fixed-step backend that same function also fills `soundSpSq`, which enters
  `falseVacuumVolumeGrowthRate` and so the percolation integral itself, so that backend's
  percolation moves slightly. It is evaluated on the symmetric phase, where there is almost no
  vacuum offset to cancel and therefore almost nothing to fix: measured on the conformal dark
  U(1) over temperatures from `1e-1` to `1e-7` of the scale, the change is at most `1.9e-6`
  relative.

  On the example point of `examples/example_point.yaml` the effect of all of this is nothing
  visible: all 54 entries of the transition table and all 58 of the observability table are
  unchanged to the digits the output carries, and the only new columns are the four
  diagnostics below. The point is healthy, which is what that shows: `DIAG:c_s_step_change`
  is `4.9e-7` against its `0.05` alarm and `DIAG:daisy_over_radiation` is `0.32`.

  The cancellation-free daisy evaluation costs time, because it decides per mode which form to
  use. A single scalar `Vtot` goes from 132 to 173 microseconds, 31% more, of which the
  analytic Debye masses are about a quarter; the two cases that actually occur, all modes small
  and none small, are each taken in one array operation rather than through the masked general
  path. End to end the cost is far smaller, because a scan is dominated by bounce solving
  rather than by potential evaluations: the example point goes from 121 to 124 seconds, under
  3%, on two concurrent runs of the same node.

- **The wall velocity takes its sound speed from the thermal part too.**
  `Hydrodynamics.calcWallVelocityLTE` is a fourth copy of the same quantity and is on the
  default path: `GWConf.wall_velocity` is `"LTE"`, and in the observables stage
  `_ensure_wall_velocity` runs before `_ensure_sound_speed`, so it was evaluated first on
  every default run and the refusal above could not protect it. Its enthalpy `-T dV/dT` and
  its two sound speeds all came from the whole potential, with the same cancellation.

  Measured on the conformal dark U(1), 31 couplings from `g = 0.45` to `0.75` and five
  temperatures from `1e-1` to `1e-7` of the scale: the broken-phase sound speed it used
  disagreed with the thermal part by more than one per cent at 42 of 135 points, and at two of
  them it was superluminal, `c_s = 1.26` at `g = 0.650` and `1.11` at `g = 0.750`, both at a
  temperature of `1e-5` of the scale. Those values went into `DTheta`, which divides by the
  sound speed, and into `find_vw`, which compares the wall velocity against it. There was no
  check that they were below one.

  `generic_potential.V_thermal` and its two derivatives now take `include_decoupled`, because
  this caller follows `GWConf.coupled_hydrodynamics` rather than always excluding the
  decoupled bath: an enthalpy taken on one plasma and combined with a pressure taken on
  another is not a thermodynamic identity. A sound speed of one or more is refused here on the
  same footing as in `calc_cs`. A phase with no plasma at all is a different statement and is
  still the runaway branch, `v_wall = 1`, as before.

  A second derivative that underflows against a first derivative that has not is a third case,
  and it is an infinite sound speed rather than either of those. It is refused in both places,
  whether the denominator reached zero exactly or is merely subnormal so that the quotient
  overflows. The comparison against one carries no finiteness test, so positive infinity
  reaches it, while `nan` and negative infinity do not compare greater than or equal to one
  and still fall through to the frozen-phase branch. In the wall velocity the sign the
  underflowed denominator kept is what decides between the two, so where it is exactly zero
  the quotient is formed rather than assumed, and that check runs before the zero-denominator
  return that would otherwise give the runaway answer first; where it is subnormal the
  quotient overflows on its own and the guard on the computed sound speeds tests positivity
  alone, which lets positive infinity through to the refusal and keeps the other two out.

  A refusal raised by the wall velocity carries the same two diagnostics as one raised by the
  sound speed. That stage runs before the sound-speed stage in both backends, so without it a
  default run would get its only refusal with neither of them, and with the least to say of
  any refusal in this change.

  The wall velocity itself barely moves, because the superluminal values were being handed to
  a `find_vw` that saturates: over the same 155 evaluations, 152 are bit-identical, the largest
  change is `1.0e-6` relative, and nothing is refused, since the superluminal values existed
  only in the route that no longer runs.

- **A sound speed of one or more is refused rather than replaced.** `c_s` sets the speed at
  which the bubbles grow, `max(v_wall, c_s)`, the efficiency factors and the spectrum, and for
  a runaway wall a superluminal value multiplies `(beta/H)_RH` by `c_s` while the spectrum
  divides by `v_wall - c_s`. `errors.SuperluminalSoundSpeedError` now names the value and the
  temperature, and suggests `GWconfig.sound_speed = "1/3"` for a run that wants the radiation
  value. Nothing is substituted: which thermodynamics a model has is the user's choice, and a
  number the model did not give would hide the fact that the potential, as evaluated, has no
  sound speed there. A phase that has simply frozen out, where the ratio is zero over zero or
  negative, is still reported as not-a-number as before. Both phases are refused: the
  symmetric-phase value is what the hydrodynamic matching across the wall is solved with.

  The refusal is not hypothetical. Over 366 evaluations spanning `g = 0.45` to `0.75` and
  temperatures from `1e-1` to `1e-9` of the scale, the released version returns a number at
  222 of them and **three of those are at or above the speed of light**: `c_s = 1.4387` at
  `g = 0.455`, `1.4799` at `g = 0.470` and `1.0536` at `g = 0.710`, all at a temperature of
  `1e-5` of the scale. With the numerics above it returns a number at all 366, none of them
  reaches one, and they lie between `0.5677` and `0.7071`. Each half of that comparison was
  computed by the version it describes, in its own source tree, through that version's own
  `Hydrodynamics.calc_cs`. That scan is the conformal dark U(1) of
  `models/TL_conformal_dark_u1.py` at `v = 0.1 GeV` and `y = 0.01`, with the sound speed taken
  on the traced broken phase at fixed field value; percolation is not computed.

  The error carries the value, the temperature, the phase it came from, CLI error code 18, and
  the two diagnostics below, so the message says which of the two known mechanisms produced
  the value rather than only that it was refused. The remedy it recommends,
  `GWconfig.sound_speed = "1/3"`, is now accepted by both percolation backends; the
  fixed-step one read only a bare number and the adaptive one only the label, so the
  recommendation was rejected by one of the two. Both resolve the setting through
  `hydrodynamics.resolve_configured_sound_speed`, which reads `"1/3"` as the massless value
  `1/sqrt(3)` and anything else as `c_s` itself, strictly between zero and one.

  On the shipped line scan of `examples/example_line.yaml`, 30 couplings from `g = 0.5` to `1.0`
  at `v = 0.14 GeV`, nothing changes status: the same two points fail with the same error code and
  the same 28 give a sound speed, none refused. Of the 79 shared columns 32 are bit-identical and
  the rest move by at most `3.2e-4` relative, in `alpha` and `alpha_thetabar`; `c_s` by `2.4e-4`,
  `betaH_RH` and `R_*` by `2.0e-4`, and the signal-to-noise ratios by about two parts in ten
  thousand. The shifts are largest at the most supercooled surviving point, where the daisy term is
  30 times the radiation, and fall away as that ratio does. The solver is deterministic: two runs
  of the same code agree bit for bit on 82 of 83 files.

- **Two diagnostics of whether the sound speed means anything.**
  `WARNING:noisy_c_s` and `DIAG:c_s_step_change` say whether it survives a change of the
  derivative step, because a value can be round-off and still be a speed, so no test on the
  value itself can find it. The step is varied both ways, by a factor ten in each direction,
  and the larger of the two changes is reported: a coarser step adds truncation error and a
  finer one adds round-off, and only the second is the failure being looked for, so testing one
  side alone would miss a value that breaks down as the step shrinks. The diagnostic does not
  propagate the refusal above. Each step is evaluated on its own and a refused value is
  compared like any other, so the three cases come apart: refused at one step and not at
  another is noise, because the value stopped being a speed somewhere inside the step range
  and no single step shows that; refused everywhere but moving is noise too; and refused
  everywhere with the same value is not, because it is not a speed but it does not depend on
  the step, and the refusal already says what is wrong with it. Catching the refusal around
  the group rather than per step let one at the base step end the measurement before either
  varied step was taken, and a constant superluminal value was then reported as maximally
  step dependent.

  Measured through `Hydrodynamics.sound_speed_is_step_dependent` on the traced broken phase,
  at `g = 0.455`, `0.7` and `0.95` and six temperatures each: with the step rule above the
  change in a healthy regime is `1.4e-7` to `9.9e-7`, a factor 50000 below the `0.05` alarm.
  With the step rule these derivatives used to inherit it runs from `2.1e-4` to `2.5e-2`, so
  at its worst a factor two below the alarm, close enough to raise it on a point with nothing
  wrong with it. `WARNING:daisy_outside_validity` and `DIAG:daisy_over_radiation`
  say whether the daisy resummation is being used where it does not apply. They report only
  on `daisy = "ArnoldEspinosa"`; with `"Parwani"` or `"off"` there is no such term in the
  potential, and both come back as a false flag beside a `nan`.

  The second is a statement about the model rather than the arithmetic. The Arnold-Espinosa
  term resums the bosonic zero Matsubara mode, which is justified where that mode is infrared
  enhanced, `m << T`. Where `m >> T` a mode should be Boltzmann suppressed, and the one-loop
  thermal integral is: it falls below what a double can represent. The daisy term does not, tending instead to
  `-(T^3/8 pi) sum n_i c_i m_i`, a power law with no `exp(-m/T)` in it. On the conformal dark
  U(1) it is 7% of the field-independent radiation at a temperature of `1e-1` of the scale,
  thirteen times it at `1e-3` and ten million times it at `1e-9`, while the lightest mode runs
  from `m/T = 1.3` to `1.3e8`. Where it dominates the free energy goes as `T^3` instead of
  `T^4`, so the entropy goes as `T^2` and `c_s^2 = s/(T ds/dT)` tends to `1/2`. No free gas at
  zero chemical potential does that: there `c_s^2 = 1/3` for massless species and masses only
  lower it. A sound speed above
  `1/sqrt(3)` in a cold, heavily supercooled phase is therefore the prescription and not the
  plasma, and the physically expected value there is the `1/sqrt(3)` of the radiation that is
  still relativistic. The flag is raised when the daisy term exceeds that radiation and the
  lightest mode has `m/T > 1`. The lightest mode is taken over every boson that contributes to
  the daisy term, which is every one with degrees of freedom, a mass squared that is not
  negative, and a Debye mass. All three are needed. A massless mode with a Debye mass is the
  lightest there is and counts: excluding it left a symmetric phase, where every
  zero-temperature mass vanishes, with no modes at all, and the ratio was discarded along with
  them in exactly the case a symmetric-phase refusal would want it. On the conformal dark U(1)
  the symmetric phase now reports ratios of `0.005` to `0.01` where it reported `nan`. A mode
  with no Debye mass does not count, however many degrees of freedom it has, because it adds
  exactly nothing to the term being judged: the transverse gauge bosons are massless and
  thermally uncorrected in every model here, and counting them pinned the lightest mass at zero
  and left the flag unable to fire at all. Whether a mode has a Debye mass is asked at the
  model's own scale, since it is a property of the mode and at the temperatures this diagnostic
  exists for the difference of the two spectra has underflowed.

  This is reported and not corrected. A resummation that is Boltzmann suppressed at low
  temperature would remove the artefact and is the subject of planned follow-up work; it is a
  change to the thermodynamics rather than a guard, and it belongs in a change of its own.

- **A timed-out point no longer comes back as a number**: `g_eff_DS` and `h_eff_DS`
  caught `BaseException` around the phase evaluation and substituted the `T = 0`
  vev, which absorbed the run's own `Timeout` and turned a timed-out point into a
  finite entropy. They are the innermost evaluations on the counted route, below the
  handlers in the percolation history, so the timeout is now re-raised in both copies
  of both functions, and the fallback for ordinary failures is unchanged. The same held
  one level up, where `percolation_temperature_from_ode` answered every exception with
  `None`: the adaptive refinement then kept the interpolated percolation temperature and
  finished a run that had timed out. Every consumer of the time-temperature factors now
  has a test that a timeout in the innermost entropy evaluation reaches the caller.

- **An unstable action slope is now retried at a tighter path deformation.**
  `calc_betaH_S3` fits the slope of `S3/T` over the support samples nearest the
  percolation temperature and repeats the fit over a smaller subset; where the two
  disagree by more than `betaH_S3_fit_rel_tol` the result was reported through
  `WARNING:betaH_S3_fit_unstable` and otherwise left alone. At the default path
  deformation the actions themselves are the reason: on a 2HDM benchmark `S3/T`
  scatters by 0.5 about a straight line across the fitting window, against a physical
  variation of 0.2 over the same window, so the slope and even its sign are not
  determined. Tightening the deformation to `converge_0 = 1` and `fRatioConv = 5e-3`,
  the two settings the `tunneltight` precision mode and the rate-jitter rescue already
  use, leaves a scatter of 0.003 and a smooth, monotonic `S3/T`. Where the two fits
  disagree the slope is therefore fitted again on a stencil of its own, spaced as the
  sparse-support fallback spaces one and computed with the deformation tightened. The
  support samples themselves are not reused: they can sit as little as `1e-6 T` apart,
  and over a baseline that short even an exact action leaves the slope undetermined, so
  recomputing them would cure noisy actions but not a stencil too short to differentiate
  over. On the benchmark above this turns `-27.3` into `+66.5`, against `+67.0` to
  `+67.3` from fitting resolved actions over the windows the solver allows. The original
  value is kept if the recomputation fails. The new setting
  `betaH_S3_fit_rescue`, with the run-time override `percolation_betaH_S3_fit_rescue`,
  switches it off.


- **Symmetric phase at extreme supercooling.** Where only the thermal masses hold the
  symmetric minimum in place, as in a classically conformal model, the potential is flat
  near the origin to within the tracing tolerance: `fmin` returns any starting point within
  about 20 `xeps` of the origin unchanged, the step predictor evaluated off the minimum
  pushes the trace further out at each step, and the last node, at the lowest tracing
  temperature, is an extrapolation that is not a minimum. The spline through those nodes
  then puts the false vacuum at an arbitrary field value over the lowest decades in
  temperature, which is where a strongly supercooled transition percolates, and everything
  read off the false vacuum there inherits the error: the masses of the fields, the counted
  entropy of the transitioning sector (`bubbledynamics.h_eff_DS`), the scale factor of the
  percolation integral and the mean bubble separation. A phase that coincides with its image
  under a symmetry of the potential at every traced temperature, within the tracer's own
  same-point tolerance, is now projected onto the fixed subspace of that symmetry
  (`phases.symmetrizeInvariantPhases`). Broken phases, phases that differ from their image
  anywhere, and field components even under the symmetry are untouched. On a drifted trace
  the counted entropy of the false vacuum falls from its 7.5 light modes to 2.1 between 30
  and 0.3 keV, and the sound speed of the symmetric phase comes out `c_s^2 = 0.324` instead
  of `1/3`. How much that costs grows with the supercooling: in the conformal dark U(1)
  shipped here nothing that percolates moves by a tenth of a per cent, the largest change
  being 1.4e-5 at a percolation temperature of 2.7 MeV and 8e-4 at 40 keV. Where the origin
  is held more weakly, as in a classically conformal U(1) without fermions, the same drift
  moved the percolation temperature by about 4 % and the mean bubble separation by about
  8 % at a percolation temperature of 2.8 keV.

- **Mirror of a phase that is its own image.** `generateMirrorPhases` compared each image
  it built only with the other images of the same phase, never with the phase it came
  from, so a phase lying on the fixed subspace of a symmetry received a second copy of
  itself. With `gen_mirror_phases = True` that second copy carries the same highest
  temperature as the original and the tracer rejects the pair, which made the classically
  conformal dark U(1) unusable with that option. Each image is now compared with its source
  as well, and a phase whose images all coincide with it contributes none. That comparison
  runs over the whole trace rather than its two ends, since dropping an image removes a
  phase outright and a trace that meets its image at both ends may still leave it in
  between. Broken phases keep their mirrors as before.

  The transition links these images carry were rebuilt with them. They named the image of
  the linked phase with the index of the transformation rather than the index plus one,
  which is the suffix the image itself is given, so every such link pointed at a phase that
  was never created; the links across the upper boundary were built from the key of a lower
  one, and raised `UnboundLocalError` for a phase with an upper link and no lower one; and a
  link to a phase that turns out to be its own image now names that phase, by the key it
  has rather than a string of it, since the links are read with `in` against the key of a
  phase and a traced phase has an integer key. Which image of which phase is kept is now
  settled for every phase before any link is built, so that a link naming an image dropped
  as a copy of another names the one kept instead, and each link is set both ways round as
  it is for traced phases. Two images count as the same one where they stay together at
  every traced temperature rather than only at the two ends of the trace: images of one
  phase under different transformations can meet at the ends and part in between, and are
  then different phases. Whether a phase lies on the fixed subspace at all is asked at the
  tolerance the symmetric phases were put there with, rather than at `diftol`: a broken
  phase can sit nearer to the subspace than two phases have to be to merge, and its image
  is then a phase of its own rather than a copy. Nothing of this is reachable with the default
  `gen_mirror_phases = False`.

- **A broken phase with no thermal pressure**: at percolation temperatures far
  below the mass scale, the thermal part of the potential underflows in the
  broken phase, so its enthalpy and its sound speed come back as zero or as not
  a number. Two places carried that forward. The wall velocity in local thermal
  equilibrium formed the sound speed as `(dV/dT)/(T d2V/dT2)`, i.e. zero over zero,
  and divided by it while building the transition strength; with the strength then
  not a number, the bounds it is compared against are passed, and the plasma was
  integrated from a non-finite initial state, raising an error from the
  integrator. The wall velocity now returns one as soon as either enthalpy or
  either second temperature derivative is unusable, since a broken phase without a
  plasma gives the wall nothing to push against, and again if either sound speed
  itself is unusable, which a subnormal first derivative against a large second
  one can cause while both ingredients still look healthy; the same check also
  guards the matching-condition solver when it is called directly. The pseudo-trace
  strengths `alpha_thetabar` and `alpha_hyd` divided by the broken-phase sound
  speed and raised a division by zero; they are now not a number where no sound
  speed exists, with the reason reported in verbose mode.
  The two sound speeds are checked separately, because only the broken-phase one
  builds the pseudo-trace while the symmetric-phase one appears solely in the
  normalisation of `alpha_thetabar`: a usable broken-phase value still gives the
  hydrodynamic strengths.
  The strengths that need no sound speed, `alpha_p`, `alpha_theta`, `alpha_e`,
  `alpha_inf` and `alpha_eq`, are unaffected. The fixed step size solver has its
  own copy of that calculation, where the same division produced an infinity and
  then a not-a-number together with a floating-point warning rather than an
  error, and where the pseudo-trace strength is called `alpha_theta`; it is
  guarded in the same way. That solver's own `calcSoundSpeedSq` also divided by
  `T d2V/dT2` unprotected, so it raised or warned before any caller could check
  the result; it now returns the not-a-number, as the adaptive solver's copy
  already did, and leaves the decision to its callers. `Hydrodynamics.calc_cs`,
  which fills the reported sound-speed columns on the default
  `gwConf.sound_speed = "compute"` path, took the square root of the same ratio
  without protection; it now returns not a number both where the ratio is zero
  over zero and where it is negative, so that an unphysical sound speed appears
  as such in the output instead of stopping the run. No result changes where the
  broken-phase plasma exists.
- **Documentation of `logGamma`**: both copies, in `bubbledynamics` and in
  `bubbledynamics_fixedstep`, said the function returns a base-ten logarithm. They
  return the natural one, `4 ln T + (3/2) ln(S/2 pi T) - S/T`, which is what every
  caller uses. Only the sentence was wrong; no behaviour changes.
- **Double percolation integral on coarse support grids**
  (`integral_method = "double_integral"`): `percIntegral` applied the
  trapezoidal rule in temperature directly to the support points of the
  adaptive solver, which leave gaps of up to several e-folds of temperature on
  the hot shoulder of the rate. Across such a gap the rule overestimates an
  exponentially falling source by about `beta du/2`, and the bubbles nucleated
  there enter the integral with the cube of their radius. On grids the solver
  produced for strongly supercooled points, the integral at the percolation
  temperature came out 40 to 60 % high and `Tperc` 1 to 4 % high, which moved
  the mean bubble separation and `betaH_RH` by up to 15 %. The integral is now
  evaluated on a sub-grid that resolves the source to a change of 0.1 in its
  logarithm, interpolating `S3/T`, `ln H`, `3 c_s^2` and `ln a` between the
  support points with a cubic spline, and the radius is accumulated from
  neighbouring scale-factor ratios in linear time. Against the closed form for
  an exponential rate in de Sitter space and against quadrature for a Gaussian
  turnover, the double integral now agrees to 0.5 % on grids with a gap of
  0.7 e-folds, where it was off by 60 to 380 %. The differential equation
  method was not affected. Two quantities are interpolated in linear space rather
  than logarithmically, `S3/T` and `3 c_s^2`, and a cubic spline through them can
  leave the range of the two support values it sits between where a feature is
  unresolved, which for a positive quantity would mean crossing zero; both are now
  held between those two values, which on a resolved interval barely binds: the
  benchmarks above are unchanged to the digits quoted and the mean bubble separation
  of a conformal dark U(1) point moves by four parts in ten million. The cap
  on sub-steps per support interval bounds the memory one interval can ask for, and
  now warns, naming the resolution actually achieved, on the rare grid where it is
  what limits that resolution instead of the requested step. A constant `3 c_s^2`
  may again be passed as a single number rather than an array, which the sub-grid
  had started to require. Where a feature of the rate lives entirely between two
  support points, the answer is set by the spacing of those points and not by the
  sub-grid between them; tests pin that, so that the limitation is not mistaken for
  one of the quadrature.

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
