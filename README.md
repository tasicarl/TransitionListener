# TransitionListener: uncertainties branch

This branch of TransitionListener accompanies the paper

> C. Tasillo, *Hubble, bubble, toil and trouble: Uncertainties in the nanohertz signal of a classically conformal
> dark U(1)*, arXiv:XXXX.XXXXX

and contains the code and the scripts that produce every figure, table and number of that work. It builds on the
public release v2.2.1 (branch `main`). The version used in the paper is archived on Zenodo,
[doi.org/10.5281/zenodo.24516038](https://doi.org/10.5281/zenodo.24516038), where every later version is listed too.

<p align="center">
  <img src="https://raw.githubusercontent.com/tasicarl/TransitionListener/main/src/transitionlistener/logo/TL-logo_large.png"
   alt="TransitionListener Logo" width="200"/>
</p>

<p align="center">
  <a href="https://www.gnu.org/licenses/gpl-3.0">
    <img src="https://img.shields.io/badge/license-GPLv3-blue.svg" alt="License: GPL v3">
  </a>
  <a href="https://arxiv.org/abs/2605.15259">
    <img src="https://img.shields.io/badge/arXiv-2605.15259-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://arxiv.org/abs/2109.06208">
    <img src="https://img.shields.io/badge/arXiv-2109.06208-b31b1b.svg" alt="arXiv">
  </a>
</p>

---

## What is new with respect to v2.2

- **Nucleation rate of the three-dimensional effective theory at next-to-leading order.** The classically conformal
  U(1) with and without two Weyl fermions (`models/TL_conformal_u1_fermions_3d.py`,
  `models/TL_conformal_u1_no_fermions_3d.py`) compute the rate from the dimensionally reduced theory, including the
  fluctuation determinants and the operator of dimension six. Each run reports whether the 3D theory is valid at its
  percolation temperature.
- **Tables of the NLO correction shipped with the package.** The functional determinants are computed once, on a
  grid of couplings and as a function of the leading-order action, which depends only on the couplings and on
  T/mu_0. One table therefore holds for every vacuum expectation value v (`transitionlistener.eft_rate.nlo_tables`, command
  `tl-nlo-table` to build tables for other settings or models).
- **About half a second per point.** One point of the conformal U(1), from the potential to the gravitational-wave
  spectrum, takes about 0.5 s on one core. v2.2.1 with the four-dimensional daisy-resummed rate takes 7 to 8 minutes
  in the region preferred by the PTA data. The percolation integral, the expansion history, the root searches and
  the bounce solver are compiled with numba, the leading-order 3D bounce comes from a shipped table, and one-field
  and multi-field models get compiled potentials.
- **The modelling choices of the uncertainty study are switches.** The source template of the gravitational-wave
  spectrum, the suppression by the cosmic expansion, the shell types, the causal tail, the speed used in the
  expansion factors, the percolation fraction, the order of the rate and its matching scale, and the PTA data set are
  set per run. `examples/example_point_3d.yaml` lists all of them. The defaults give the spectrum of v2.2, the
  template of the LISA Cosmology Working Group (arXiv:2403.03723) without the suppression by the expansion.

The full list is in [CHANGELOG.md](CHANGELOG.md).

## Installation

```bash
git clone -b uncertainties https://github.com/tasicarl/TransitionListener.git
cd TransitionListener
pip install -e .
```

Runs on the shipped NLO tables need nothing else. Building new tables needs BubbleDet: `pip install -e ".[nlo]"`.
The binary foreground of the paper uses [binaryfog](https://pypi.org/project/binaryfog/) 1.0.0
(`pip install binaryfog==1.0.0`).

## First run

```bash
tl -c examples/example_point_3d.yaml -v
```

computes one point of the conformal U(1) without fermions (g = 0.45, v = 2.9 GeV) with the defaults of the paper, in
about ten seconds including the start of the interpreter. The file writes out every switch of the uncertainty study
with its alternatives and the label of the variant in the paper.

## Reproducing the paper

[paper/README.md](paper/README.md) describes the two stages: `paper/figs/make_figures.py` rebuilds all figures and
tables from the shipped results in `paper/data` in a few minutes, and the drivers in `paper/reproduce/jobs` recompute
those results.

## Using the model with fermions inside your own scan code

A scan framework that computes its own likelihood (for example GAMBIT) calls TransitionListener point by point through
`transitionlistener.spectrum.gw_spectrum`. It takes the model file, the model's parameters and every switch of the
uncertainty study, and returns the gravitational-wave spectrum today as a function of frequency, together with the
transition parameters it was computed from:

```python
from transitionlistener.spectrum import gw_spectrum

h2Omega = gw_spectrum(
    "models/TL_conformal_u1_fermions_3d.py",
    g=0.5, y=0.3, v_GeV=1.0,                                   # gauge coupling, Yukawa coupling, vev in GeV
    nucleation=dict(                                           # the rate: 3D theory at NLO, as in the paper
        include_determinants=True, include_dim6=True, mu4_over_T=7.0553,
        matching_order="NLO", temporal_coupling="h3", input_scheme="zero_quartic", mu3_choice="gT",
        nlo_grid_file=None),                                   # None: the shipped table
                                                               # src/transitionlistener/tab_data/nlo_grids/conformal_u1_fermions.npz
    percolation_f_perc=0.28957,                                # true-vacuum fraction at percolation
    gw_source_model="LewickiVaskonen22", gw_expansion_suppression=True,
    gw_lv22_expanding_shape="dissipative", gw_causality_tail="smooth",
    gw_expansion_speed="betaH_Gamma", gw_bw_collisions="full",
    gw_use_mean_bubble_separation=True,                        # the spectrum
)
h2Omega(f_Hz)                    # h^2 Omega_GW at the frequencies f_Hz (array, in Hz)
h2Omega.observables              # Tperc, Treh_SM_GeV, alpha, betaH_RH, betaH_Gamma, betaH_S3, ...
```

A point without a completed transition raises the error of the pipeline; its `errorcode` says why (see the error
reference of the manual). One point takes about a second once the interpreter runs. Points to keep in mind:

- The shipped NLO table of the model with fermions covers 0.38 <= g <= 1 and 0 <= y <= 1. Outside it, or at a
  temperature outside the range of the table where the rate matters, the run stops with `NLOGridError`.
- `mu4_over_T` must be one of the three values of the table, 3.52765, 7.0553 or 14.1106. `matching_order`,
  `input_scheme`, `mu3_choice` and `temporal_coupling` must keep the values above unless a table built with
  `tl-nlo-table` for other settings is attached through `nlo_grid_file`.
- The four-dimensional rates of the study are other model files: `models/TL_conformal_u1_fermions_rg.py` (the
  couplings run to pi T, variant N4, from a shipped table of its action in the same range of g and y) and
  `models/TL_conformal_u1_fermions.py` (fixed couplings, N5), both without the `nucleation` argument.
- `timeout` (seconds) uses `signal.alarm` and works only in the main thread of the process.
- All switches and their alternatives are listed in `examples/example_point_3d.yaml`.

## Citation

If you use this branch, please cite the paper above, the [v2 release](https://arxiv.org/abs/2605.15259), the
original [v1 release](https://arxiv.org/abs/2109.06208) and the underlying
[`CosmoTransitions`](https://arxiv.org/abs/1109.4189). The matching relations of the 3D theory were derived with
[`DRalgo`](https://arxiv.org/abs/2205.08815), and the functional determinants are computed with
[`BubbleDet`](https://arxiv.org/abs/2308.15652). In case you use the LTE wall velocity approximation
(`src/transitionlistener/hydrodynamics.py`), please also cite [arXiv:2303.10171](https://arxiv.org/abs/2303.10171).

## License

TransitionListener is distributed under the GNU GPL v3.0 license. Derivative works must remain open source under
the same license and credit the original authors. See [LICENSE](LICENSE.txt).
