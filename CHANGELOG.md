# Changelog

All notable changes between releases of TransitionListener.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

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

- **A sound speed that is not a speed no longer sets the expansion speed of the bubbles.**
  `c_s` is taken as `c_s^2 = (dV/dT)/(T d2V/dT2)` by finite differences of the full
  potential. Under extreme supercooling the thermal part of the potential sits below double
  precision against the vacuum energy, `Tperc^4/Delta V` reaching `1e-16` to `1e-20`, and
  those differences return round-off; the cases that come out as zero over zero or negative
  were already turned into not-a-number, but a finite round-off value was not, and values
  above one were seen. Since the bubbles grow at `max(v_wall, c_s)`, such a value replaces
  the wall velocity, and for a runaway wall it inflates `(beta/H)_RH` by the factor `c_s`.
  On a classically conformal dark U(1) scan reaching percolation temperatures of `1e-8` of
  the scale, 7 of the 264 points with a finite sound speed came out between 1.01 and 1.71,
  and `(beta/H)_RH` there was 2.3% to 69.5% above a direct integration of the false-vacuum
  fraction while `Tperc`, which does not use the sound speed, agreed to 0.8%.

  `hydrodynamics.physical_sound_speed` now replaces a value that is not a speed, meaning
  not-a-number, infinite, non-positive or `c_s >= 1`, with the massless value `1/sqrt(3)`,
  in both solvers. `c_s_bro` keeps the computed number, so the output still shows what the
  potential gave, and `WARNING:unphysical_c_s` says when the replacement happened.

  **Only values that are not speeds are replaced, and that is deliberate.** A plasma whose
  particles have masses has `c_s^2 < 1/3`, reaching `1/3` only in the massless limit and from
  below, as the lattice equation of state of quantum chromodynamics shows (arXiv:1309.5258,
  arXiv:1407.6387). Clamping at `1/sqrt(3)` would therefore be the physical bound, and it is
  left to a change of its own for two reasons.

  It fires on 108 of 286 campaign runs, and on the example point of
  `examples/example_point.yaml`, whose broken-phase sound speed is 0.589508, 2.1% above
  `1/sqrt(3)`. That is not stencil noise: it is stable to six digits over temperature steps
  from `dT/T = 1e-3` to `1e-1`, so it is a property of the effective potential, and clamping
  would override a converged number. And on that point the clamp moves the spectrum, by
  -2.7% in the sound-wave peak frequency, +0.4% in the peak amplitude, +3.4% in one
  pulsar-timing signal-to-noise ratio, and one detectability verdict from false to true. The
  efficiency factors do not move there, because `v_wall = 1` exceeds the sound speed either
  way; they do depend on it for a wall slower than the sound speed. With the guard as it
  stands the example point is identical in every observable with it switched off, the wall
  clock aside.

  **What the guard does not do.** It is not a detector of the round-off regime: a value can be
  round-off and still be a speed. On the conformal dark U(1) at a temperature where
  `Tperc^4/Delta V` is 2.8e-16 the computed `c_s` is 0.82 and swings over `c_s^2 = -0.001` to
  `0.89` with the derivative step, and passes untouched. Catching that needs the temperature
  derivatives of the thermal part of the potential alone, without the vacuum energy that
  cancels in them, which is a change to the thermodynamics rather than a guard.

  Two other computations of the same quantity keep the raw value, deliberately and for now.
  `bubbledynamics.calcSoundSpeedSq`, which builds the pseudo-trace and `alpha_hyd`, and
  `Hydrodynamics.calcWallVelocityLTE`, which is the default wall velocity, each check only
  that the value is finite and positive. Routing them through this guard would change
  `alpha_hyd` by up to a factor three on the points that motivate this entry, so it is a
  change to the energy budget rather than a guard on a speed, and wants its own measurement.

  The not-a-number case is replaced as well, not only the finite round-off values. For a
  runaway wall that changes nothing, since `max(v_wall, nan)` already returned the wall
  velocity, but a run with a configured wall slower than `1/sqrt(3)` now gets the radiation
  value where it previously had none, which raises `(beta/H)_RH` on exactly the points that
  had no usable sound speed.

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
