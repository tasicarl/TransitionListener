# Reproducing the paper

"Hubble, bubble, toil and trouble: Uncertainties in the nanohertz signal of a classically conformal dark U(1)".

Every figure, table and number of the paper (arXiv:XXXX.XXXXX) is produced by the scripts in this folder:

* **Stage A** (`figs/`) builds every figure and every `\input` table from the results shipped in `data/`, in about a
  minute when the cached figure fields in `figs/figure_data/` are current and in a few minutes otherwise.
* **Stage B** (`reproduce/`) recomputes the results in `data/` from TransitionListener and the public pulsar-timing
  likelihoods. It needs a cluster.
* **The numbers** quoted in the text are printed by `reproduce/numbers/*.py` from `data/`, into
  `reproduce/numbers/output/`.

```
paper/
  budget_table.tex switch_numbers.tex switch_numbers_ipta.tex   tables of the paper, written by stage A
  figs/        make_figures.py (stage A), make_schematics.py, budget_table.py, figure_checks.py, the figures,
               figure_data/ (cached fields of the figures, keyed by a digest of their inputs)
  data/        every input of stage A and of the number scripts
  reproduce/   stage B: linescan/ rate/ daisy/ plane/ binaries/ numbers/, jobs/ (one driver per step), paths.py
```

## Environment

* Python 3.10 with the dependencies of TransitionListener (this repository: `PYTHONPATH=<repo>/src`), PTArcade and
  ceffyl, and binaryfog 1.0.0 (`pip install binaryfog==1.0.0`). binaryfog fetches the public NANOGrav and IPTA free
  spectra on first use.
* holodeck, only for `reproduce/binaries/binaries_prior_band_holodeck.py` (the binary band of figure 3, stage B).

## Stage A: from data/ to the figures and tables

From `paper/`, with `PYTHONPATH=<repo>/src`:

```
(cd figs && python make_figures.py)     # fig_pipeline, fig_rate, fig_spectra, fig_plane, fig_hardening, fig_budget,
                                        # fig_foreground, switch_numbers.tex, switch_numbers_ipta.tex, budget_table.tex
```

`make_figures.py name ...` builds single targets. A missing input stops the build with `FileNotFoundError`; the build
also fails if a requested output was not written in the run or a figure does not have the width at which the paper
includes it. Two kinds of fields are computed rather than read: the transition table evaluated on the fine grids of
figures 4 and 7 (`reproduce/plane/transition_table.py`) and the spectra of figure 3 (`reproduce/plane/plane_scan.py`).
They are cached in `figs/figure_data/` under a digest of every file they depend on, TransitionListener's sources
included, so a change of any input recomputes them (4 min on a 112-core node).

## Stage B: the results in data/

Each driver in `reproduce/jobs/` runs one step; the machine-specific settings are environment variables (`PY`, the
interpreter; `NCPU`; `TL_CODE_ROOT`; `PAPER_WORK`, the folder of intermediate results, by default `reproduce/work/`),
documented in `reproduce/jobs/env.sh`. Under SLURM, `sbatch --cpus-per-task=56 jobs/<step>.sh`. In order:

| step | writes (in `data/` unless stated) | scripts | cost |
|---|---|---|---|
| `01_linescan.sh` | `work/linescan_points/` (`LINESCAN_POINTS`), `linescan_u1_mu0_1GeV.json`, the three points of `linescan_points/` | `linescan/scan_point.py` (runs `pipeline_one.py`, `nlo_table_one.py`), `collect_linescan.py` | 4 min, 56 cores |
| `02_tl_line.sh` | `work/accuracy_table.json` | `linescan/capture_tl.py`, `accuracy_table.py` | 1 min, 18 cores |
| `03_actions.sh` | `action_cache/` | `rate/compute_actions.py` | about a day, 56 cores |
| `04_transition_table.sh` | `transition_table.npz`, `line_budget.json` | `rate/build_exact_table.py`, `build_falsevac_table.py`, `merge_tables.py`, `line_budget.py` | 2 h, 56 cores |
| `05_daisy.sh` | `work/daisy_cache/` (`DAISY_CACHE`), the three files of `daisy_cache/`, `daisy_table_plane*.npz`, `daisy_rg_table_plane.npz`, `daisy_line*.json`, `scale_band_g0.500.json` | `daisy/compute_daisy_actions.py`, `build_daisy_table.py`, `prepare_daisy_table.py`, `daisy_line.py`, `daisy_scale_band.py` | several hours, 56 cores |
| `06_tl_nodes.sh` | `tl_nodes_table.npz`, `kappa_nodes_table.npz` | `plane/tl_nodes.py` at the nodes of `plane/tl_nodes.txt`, `collect_tl_nodes.py` | 4 min, 56 cores |
| `07_plane.sh` | `plane_scan_u1.npz`, `best_fits.json`, `pta_free_spectra.npz`, `notail_scan.npz`, `bhb_lnl_without_prior.json` | `plane/plane_scan.py`, `best_fits.py`, `pta_free_spectra.py`, `notail_scan.py`, `binaries/bhb_lnl_without_prior.py` | 1 h, 56 cores |
| `08_binaries.sh` | `marginal_foreground.npz`, `ipta_fog*.npz`, `ng15_fog_priors.npz`, `ng15_fog_prof_recent.npz`, `bhb_population_dlnl.json`, `preference_tau0.1.json`, `preference_tau4.json`, `preference_tau_profile.npz` | `binaries/marginal_foreground.py`, `ipta_fog.py`, `ipta_fog_priors.py`, `prior_study.py`, `collect_prior_study.py`, `bhb_population_dlnl.py`, `preference.py`, `preference_map.py`, `preference_tau_profile.py` | several hours, 56 cores |
| `09_tail.sh` | `tail_check/tail_patch.npz`, `tail_fog.npz`, `tail_fog_recent.npz`, `tail_fog_prof_recent.npz` | `plane/tail_fine.py`, `binaries/tail_fog.py`, `tail_fog_plateau.py`, `tail_fog_prior.py`, `tail_fog_prof_prior.py`, `plane/merge_high.py` | 6 h, 56 cores |
| `10_holodeck_band.sh` | `binaries_prior_band_holodeck_recent.npz` | `binaries/binaries_prior_band_holodeck.py` (holodeck environment) | hours, 40 cores |
| `11_numbers.sh` | `reproduce/numbers/output/*.txt`, `determinant_reference_values.csv` | `numbers/*.py` | minutes |

Steps 01 to 06 depend on each other in this order (04 needs 02 and 03; 05 and 06 need 04); 07 needs 01 to 06; 08 and
09 need 07; 10 is independent; 11 needs everything else. `plane_scan.py` stops with an error if an input is missing,
instead of treating a coupling as one without a phase transition. The action caches, line-scan points and daisy
actions that stage A itself reads (those of g = 0.5 and of the three points of table 5) are shipped in `data/`; the
others are written there by steps 01, 03 and 05.

The shipped results were computed with the code in this folder and the TransitionListener of this repository at the
time of the computation; the tables of transition parameters and the likelihood maps were last recomputed together
for the prior range 3.2 MeV < v < 10 TeV of the scan.
