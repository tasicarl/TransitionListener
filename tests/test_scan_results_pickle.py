"""A parallel scan sends each result back to its parent, so the spectrum of a point has to survive pickling."""
import pickle
from pathlib import Path

import numpy as np
import pytest

from transitionlistener.helper_functions import load_potential
from transitionlistener.interface import _compute_single_point

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def spectrum():
    cls = load_potential(str(REPO / "models/TL_dark_U1_g_parameterization.py"), "specific_potential")
    context, error = _compute_single_point({"g": 1.0, "lambda": 0.02, "v_GeV": 0.1}, cls, timeout=-1, verbose=False,
                                           include_smbhb=False, max_stage=4)
    assert error is None
    return context["gwspectrum"]


def test_the_spectrum_of_a_point_comes_back_from_a_worker(spectrum):
    back = pickle.loads(pickle.dumps(spectrum.gwspec_dict))
    f = np.geomspace(1e-9, 1e-6, 5)
    np.testing.assert_array_equal(back["h2OmegaGW_fun"](f), spectrum.gwspec_dict["h2OmegaGW_fun"](f))


@pytest.mark.parametrize("model", ["LISA-pipeline24", "LISA-update2019", "LewickiVaskonen22"])
def test_the_tail_with_expansion_leaves_nothing_that_cannot_be_sent(spectrum, model):
    other = pickle.loads(pickle.dumps(spectrum))
    other.source_model, other.expansion_suppression = model, True
    other.sources = list(other.SOURCES[model])
    other.h2Omega_0_sum(np.geomspace(1e-9, 1e-6, 5), other.sources[0])
    pickle.dumps(other)
