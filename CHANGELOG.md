# Changelog

All notable changes between releases of TransitionListener.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [2.0.0] - 2026-05-06

Major code release accompanied by the paper
[arXiv:2605.xxxxx](https://arxiv.org/abs/2605.xxxxx) (TDB).

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
