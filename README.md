# TransitionListener

A framework for analyzing cosmological first-order phase transitions and their gravitational wave signatures.

<p align="center">
  <img src="./src/transitionlistener/logo/TL-logo_large.png" alt="TransitionListener Logo" width="200"/>
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

**TransitionListener** is an open-source Python package designed to compute, analyze, and visualize **first-order phase transitions** and their resulting **stochastic gravitational wave backgrounds**. It implements a full thermal history from microphysical parameters to observable signals, combining precision in the thermodynamic description of the primordial plasma with a large range of possibilities for parameter scans and automatically generated plots. TransitionListener bridges theoretical particle physics and gravitational wave phenomenology, enabling robust parameter inference across twelve orders of magnitude in gravitational wave freqeuncies.

Please refer to our online manual at [https://tasillo.de/TransitionListener/](https://tasillo.de/TransitionListener/) for detailed information on the installation and implementation details.

---
## Key Features
Our code in version 2 is extends C. Wainwright's ```CosmoTransitions``` (see ```arxiv:1109.4189```) and its original version 1 (used in ```arxiv:2109.06208```) in multiple ways:
- **Precision percolation computation** with self-consistent iteration over the Hubble rate and the true-vacuum fraction
- **Consistent treatment of the transition speed** and the mean bubble separation.  
- **Bubble wall velocity** modeling in local thermal equilibrium based on Ai et al.'s ```arxiv:2303.10171```
- **State-of-the-art gravitational wave spectra** including multiple source contributions, as recommended by the LISA Cosmology Working Group in ```arxiv:2403.03723```
- **Built-in sensitivity curves** for LISA, BBO, DECIGO, muAres and PTA experiments  
- **PTA log-likelihood** evaluation using ```PTArcade``` based on the ```Ceffyl``` backend.
- **UltraNest integration** for scans over large model parameter spaces using nested sampling methods.  
- **Energy density**: Evaluated self-consistently using the user-defined effective potential, going beyond the simple and often-used $\Delta V$ approximation.  
- **Stable at low temperatures** — tested up to extreme supercooling of $\alpha = 10^{10}$  
- **Robust error codes** telling the difference between numerical errors and physics reasons, indicating why a given parameter point does not yield a gravitational wave signal, even if you expected it to do so 
- **First Python code** supporting multi-Higgs potentials and SNR computation simultaneously  
- **Flexible nucleation and percolation criteria** which go far beyond the fixed $S_3/T \simeq 140$ assumption: We take the degrees of freedom of the user-defined SM extension and the amount of vacuum energy into account when checking for the nucleation and percolation of bubbles.
---

## Installation and first use

Good news: TransitionListener is very easy to install! On most machines, it is enough to just clone this repository and ```pip install``` it:

```bash
git clone https://github.com/tasicarl/TransitionListener.git
cd TransitionListener
pip install -e .
```

You're now ready to use TransitionListener on your own favourite model. Alternatively, take one of the models shipped with the package. A minimal working example is

```bash
tl -c examples/example_point.yaml
```

which reads in the YAML file shipped with the repository, computes the full phase-transition history for a benchmark point of a U(1) extension of the Standard Model, predicts the gravitational wave spectrum, and evaluates its observability with LISA, PTAs and other observatories.

More information and many more hands-on use cases of the code can be found in the manual at 

[https://tasillo.de/TransitionListener/](https://tasillo.de/TransitionListener/).

## Authors
Please feel free to write us an email in case you identify any bug in the code or still need some further documentation. Enjoy!

- Jonas Matuszak (KIT, jonas.matuszak@kit.edu)
- Carlo Tasillo (IFIC Valencia, carlo.tasillo@ific.uv.es)

## Citation

If you use TransitionListener in your research, please cite the v2.0 release paper as well as the original v1 release in the bibliography of the new paper.

## License

TransitionListener is distributed under the GNU GPL v3.0 license.
You are free to use, modify, and distribute the code — provided that derivative works remain open-source under the same license and credit the original authors.
See the [LICENSE](LICENSE.txt) file for full details.


<p align="center">
  <sub>© 2026 J. Matuszak & C. Tasillo: TransitionListener v2.0 – Robust gravitational wave predictions for cosmological phase transitions.</sub>
</p>
