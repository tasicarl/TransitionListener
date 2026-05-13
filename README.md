# TransitionListener

A framework for analyzing cosmological first-order phase transitions and their gravitational-wave signatures.

<p align="center">
  <img src="./src/transitionlistener/logo/TL-logo_large.png" alt="TransitionListener Logo" width="450"/>
</p>

---

<p align="center">
  <a href="https://www.gnu.org/licenses/gpl-3.0">
    <img src="https://img.shields.io/badge/license-GPLv3-blue.svg" alt="License: GPL v3">
  </a>
  <a href="https://johannesbuchner.github.io/UltraNest/">
    <img src="https://img.shields.io/badge/sampler-UltraNest-lightgrey.svg" alt="Sampler: UltraNest">
  </a>
  <a href="https://arxiv.org/abs/2109.06208">
    <img src="https://img.shields.io/badge/arXiv-2109.06208-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://arxiv.org/abs/2502.19478">
    <img src="https://img.shields.io/badge/arXiv-2502.19478-b31b1b.svg" alt="arXiv">
  </a>
</p>

---

## Overview

**TransitionListener** is an open-source Python package designed to compute, analyze, and visualize **first-order phase transitions** and their resulting **stochastic gravitational-wave backgrounds**. It implements a full thermal history from microphysical parameters to observable signals, combining field-theoretic precision with modern Bayesian inference tools.

TransitionListener bridges theoretical particle physics and gravitational-wave phenomenology, enabling robust parameter inference across cosmological and detector scales.

Our code in version 2 is extends C. Wainwright's ```CosmoTransitions``` (see ```arxiv:1109.4189```) and its original version 1 (used in ```arxiv:2109.06208```) in multiple ways:

## Key Features
- **Precision percolation computation** with self-consistent iteration over the Hubble rate and the false vacuum fraction
- **Consistent treatment of the transition speed** and the mean bubble separation.  
- **Bubble wall velocity** modeling in local thermal equilibrium based on Ai et al.'s ```arxiv:2303.10171```
- **State-of-the-art gravitational-wave spectra** including multiple source contributions, as recommended by the LISA Cosmology Working Group in ```arxiv:2403.03723```
- **Built-in sensitivity curves** for LISA, BBO, DECIGO, muAres and PTA experiments  
- **PTA log-likelihood** evaluation using ```PTArcade```.
- **UltraNest integration** for scans over large model parameter spaces using nested sampling methods.  
- **Automatic degrees-of-freedom accounting** from both SM and BSM sectors  
- **Energy density**: Evaluated self-consistently using the user-defined effective potential, going beyond the simple and often-used $\Delta V$ approximation.  
- **Stable at low temperatures** — handles extreme supercooling of up to $\alpha = 10^{10}$  
- **Robust error codes** which indicate why a given parameter point does not yield a gravitational wave signal, even if you expected it to do so 
- **First Python code** supporting multi-Higgs potentials and SNR computation simultaneously  
- **Flexible nucleation and percolation criteria** which go far beyond the fixed $S_3/T \simeq 140$ assumption: We take the degrees of freedom of the user-defined SM extension and the amount of vacuum energy into account when checking for the nucleation and percolation of bubbles.
- **Fully modular design** with forthcoming **GAMBIT integration**
---

## Installation

Good news: TransitionListener is very easy to install!

### Core installation
For the release smoke test and the standard gravitational-wave workflow, the core Python package is sufficient:

```bash
git clone https://github.com/tasicarl/TransitionListener.git
cd TransitionListener
pip install -e .
```

This exposes the `tl` console script in your active environment.

### Optional PTA environment
PTA likelihood evaluation uses the external `PTArcade` / `enterprise` stack, which in turn may require native libraries that are better provisioned through conda than plain `pip`.

First, install micromamba or conda on your computer or computing cluster. To create the recommended PTA-capable environment **on a Linux machine** use:

```bash
conda create -n TL -c conda-forge python=3.10 ptarcade ultranest tqdm corner getdist mpi4py
conda activate TL
```

If you're using Apple Silicon or Intel **macOS** systems, use:

```bash
conda create -n TL --platform osx-64 -c conda-forge python=3.10 ptarcade ultranest tqdm corner getdist mpi4py
conda activate TL
```

Then install TransitionListener into that environment:

```bash
git clone https://github.com/tasicarl/TransitionListener.git
cd TransitionListener
pip install -e .
```

If you want `pip` to record the optional PTA dependency explicitly, install the extra instead:

```bash
pip install -e ".[pta]"
```

### Testing
After installing, run the release smoke test (≈1 minute on a modern CPU):

```bash
pip install pytest
pytest tests/test_release_smoke.py
```

It runs `tl -c examples/example_point.yaml` end-to-end on the conformal U(1) benchmark and checks that the key observables (Tperc, Treh, alpha, RH) land in their expected physical bands.

## Quick start
You're now ready to use TransitionListener on your own favourite model. Alternatively, take one of the models shipped with the package. A minimal working example is

```bash
tl -c examples/example_point.yaml
```

which reads in the YAML file shipped with the repository, computes the full phase-transition history for a benchmark point of a U(1) extension of the Standard Model, predicts the gravitational-wave spectrum, and evaluates its observability with LISA, PTAs and other observatories.

For a full reproduction of every figure in the v2.0 paper:

```bash
python arxiv/reproducibility/paper/scripts/build_all.py            # only build what's missing
python arxiv/reproducibility/paper/scripts/build_all.py --regenerate  # rerun every scan + figure
python arxiv/reproducibility/paper/scripts/build_all.py --only label_bubble-separation
```

The figure-to-script-to-YAML mapping lives in [`arxiv/reproducibility/paper/manifest.yaml`](arxiv/reproducibility/paper/manifest.yaml).

### Environment rule
For this repository, use the `TL` conda environment for all Python-based workflows:

```bash
conda activate TL
```

This includes:
- production scans
- profiling runs
- plotting scripts
- local validation helpers
- one-off Python commands using `numpy`, `matplotlib`, or repo imports

If you prefer not to activate the environment globally, use:

```bash
conda run -n TL python ...
```

More information and many more hands-on use cases of the code can be found in the [manual](https://tasillo.de/TransitionListener/).

## Authors
Please feel free to write us an email in case you identify any bug in the code or still need some further documentation. Enjoy!

- Jonas Matuszak (KIT, jonas.matuszak@kit.edu)
- Carlo Tasillo (IFIC Valencia, carlo.tasillo@ific.uv.es)

## Citation

If you use TransitionListener in your research, please cite the v2.0 release paper as well as the original v1 release in the bibliography of the new paper. A `CITATION.cff` file is shipped with the repository so that GitHub renders a "Cite this repository" button automatically.

## License

TransitionListener is distributed under the GNU GPL v3.0 license.
You are free to use, modify, and distribute the code — provided that derivative works remain open-source under the same license and credit the original authors.
See the [LICENSE](LICENSE.txt) file for full details.


<p align="center">
  <sub>© 2026 J. Matuszak & C. Tasillo: TransitionListener v2.0. Gravitational wave backgrounds from first-order phase transitions — made simple.</sub>
</p>
