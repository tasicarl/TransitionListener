"""A rejection must not blame a resource that was never exhausted.

``_raise_if_large_delta_p_underresolved`` used to end its message with "the controller
already spent its remaining support budget on local jump refinement". That is frequently
untrue, and it sends whoever reads it to the wrong place: at lambda3 = 5.8136778037 the
point was rejected with **115 of its action evaluations still unspent**, having instead
reached the controller's limit of maxit = 10 support updates. Four separate budget-based
fixes were written and measured against that sentence before the traces showed the budget
was never the constraint.

The message now reports whichever resource actually ran out, and says nothing about a cause
when the caller does not supply the numbers.

Part of TransitionListener v2
Documentation: https://tasillo.de/TransitionListener/
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from transitionlistener import errors
from transitionlistener import percolation_adaptive_gridbuilders as gridbuilders


def settings():
    return types.SimpleNamespace(
        large_delta_p_refine_threshold=0.1, large_delta_p_success_threshold=0.2,
        n_action_increment=5, f_start=1.0e-3, f_final=0.99, maxit=10,
    )


def unresolved_profile():
    """A profile with a jump the acceptance test rejects, in the strict P < 0.2 region."""
    T = np.array([40.0, 38.0, 36.169, 35.8, 34.0, 32.0])
    P = np.array([0.0005, 0.02, 0.15568, 0.21346, 0.30, 0.45])
    return T, P


def _message(**resources):
    T, P = unresolved_profile()
    with pytest.raises(errors.PercolationError) as caught:
        gridbuilders._raise_if_large_delta_p_underresolved(
            T, P, settings(), stage="step 3", outcome="successful", **resources)
    return str(caught.value)


def test_the_jump_itself_is_still_reported():
    """Premise: the profile is rejected, and the message locates the jump."""
    message = _message()
    assert "sharp unresolved jump" in message
    assert "Delta P" in message


def test_an_unspent_budget_is_not_described_as_spent():
    """The defect: a run stopped by the iteration limit was told its budget ran out."""
    message = _message(remaining_budget=115, iterations_used=10, max_iterations=10)
    assert "115 action evaluations still unspent" in message
    assert "10 support updates" in message
    assert "budget fully spent" not in message


def test_a_spent_budget_is_reported_as_spent():
    message = _message(remaining_budget=0, iterations_used=4, max_iterations=10)
    assert "support budget fully spent" in message
    assert "still unspent" not in message


def test_budget_left_without_the_iteration_limit_says_so_too():
    """Neither resource ran out, so the message must not imply one did."""
    message = _message(remaining_budget=42, iterations_used=3, max_iterations=10)
    assert "42 action evaluations still unspent" in message
    assert "support updates" not in message


def test_no_claim_is_made_when_the_caller_does_not_know():
    """A caller without the numbers must produce no causal claim at all."""
    message = _message()
    for phrase in ("unspent", "fully spent", "support updates",
                   "already spent its remaining support budget"):
        assert phrase not in message, f"the message asserted {phrase!r} without evidence"


def _rejection_call_sites():
    """Every call to the rejection in the adaptive solver, with the keywords it passes."""
    import ast

    from transitionlistener import percolation_adaptivestepsize as adaptive

    tree = ast.parse(open(adaptive.__file__).read(), adaptive.__file__)
    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = (node.func.attr if isinstance(node.func, ast.Attribute)
                else getattr(node.func, "id", None))
        if name == "_raise_if_large_delta_p_underresolved":
            sites.append((node.lineno, {kw.arg for kw in node.keywords}))
    return sites


def test_the_solver_supplies_the_numbers_at_every_rejection():
    """The message can only name the resource the caller tells it about.

    `_resource_note` falls silent without the three numbers, which is the right default for
    a caller that does not have them -- and which also means a solver call site that stops
    passing them silently loses the whole diagnosis, with the tests above still green
    because they call the rejection directly. The four step-3 exits all have the numbers in
    hand, so all four must pass them. Driving step 3 needs a real profile and a Tperc solve,
    so the call sites are read instead.
    """
    sites = _rejection_call_sites()
    assert len(sites) == 4, f"expected four rejection call sites, found {len(sites)}"
    required = {"remaining_budget", "iterations_used", "max_iterations"}
    for lineno, keywords in sites:
        missing = sorted(required - keywords)
        assert not missing, (
            f"percolation_adaptivestepsize.py:{lineno} rejects without {missing}, so the "
            "message says nothing about which resource ran out")
