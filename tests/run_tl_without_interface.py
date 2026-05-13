from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from models.TL_dark_U1 import specific_potential
from transitionlistener.phases import Phases
from transitionlistener.transitions import Transitions
from transitionlistener.transitionObservables import TransitionObservables
from transitionlistener.observability import Observability
from transitionlistener.gwfopt import FOPTspectrum
from transitionlistener.plots import plotGWSpectrum
from transitionlistener.pathDeformation import bounceSolution
from transitionlistener.bubbledynamics import Gamma

# Initialising the potential
# =====================================================
pot = specific_potential(inp={"g_tilde": 2.69, "l": 1.5e-3, "v_GeV": 1e7})

# Phases and possible transitions
# =====================================================
phases = Phases(pot, verbose=True)
transitions = Transitions(phases, pot, verbose=True)

# Percolation
# =====================================================
observables = TransitionObservables(pot, phases, transitions, verbose=True)
percolation = observables.percolationResults[0]
Tperc = percolation.Tperc
Tperc_GeV = Tperc * pot.conversionFactor

# Bounce action, bubble profiles and , nucleation rates
# ======================================================
x_start, x_end = phases[1].valAt(Tperc), phases[0].valAt(Tperc)
bSol = bounceSolution(pot, Tperc, x_start, x_end)
action = bSol.action
Phis = bSol.Phi
R = bSol.profile1D.R
nuclRate = Gamma(Tperc, action)

# Gravitational wave spectrum
# ======================================================
gwspectrum = FOPTspectrum(observables[0], verbose=False)
observability = Observability(gwspectrum, verbose=True, include_smbhb=False)
plotGWSpectrum(observables[0], showplot=True)
