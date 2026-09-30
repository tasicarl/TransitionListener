# Instructions for code review

TransitionListener computes the thermodynamics, percolation and gravitational wave spectrum of a
cosmological first-order phase transition. Physics correctness and reproducibility matter more
than style.

## What this repository looks like

There are **two percolation solvers with parallel implementations**, and a function often exists
twice under the same name:

| adaptive step size | fixed step size |
|---|---|
| `bubbledynamics.py` | `bubbledynamics_fixedstep.py` |
| `transitionObservables.py` | `transitionObservables_fixedstep.py` |
| `percolation_adaptivestepsize.py` | `percolation_fixedstepsize.py` |

`_build_percolation_settings`, `percIntegral`, `calc_betaH_S3`, `logGamma`, `h_eff_DS`,
`calcSoundSpeedSq` and `h_eff_coupled_radiation` all exist in both trees. A module in the
fixed-step family *usually* imports from the fixed-step copy, but not always: for example
`transitionObservables_fixedstep.py` takes `falseVacuumVolumeGrowthRate` and
`integrate_broken_temperature` from the adaptive `bubbledynamics`. Check the import, do not assume
it from the file name. A change to one copy does not reach the other, and a guard added to the
wrong copy is dead code on the path it was meant to protect. **This is the most valuable thing to check in any diff**: when a function
is changed, say whether its twin needs the same change, and whether the entry point a real run
takes reaches the copy that was edited.

## Things that are deliberate, not defects

- **`transitionObservables_fixedstep.py` is usually left unchanged.** The two solvers are to be
  brought onto one implementation in a change of its own. A pull request that changes only the
  adaptive solver and says so is doing the right thing; it is not an oversight.
- **A new setting that only the adaptive solver implements raises for the fixed-step solver.**
  That is the intended behaviour, not a missing feature.
- **Diagnostics are written apart from the observables**, under a `Diagnostics:` heading in
  `1_All_params.txt`. This keeps two runs of one point byte-comparable in the observables block;
  the wall clock would otherwise break that. Do not suggest merging the two blocks.
- **`gen_mirror_phases` and `integral_method = "double_integral"` are non-default paths.** Changes
  confined to them do not affect published results, and the pull request will say so.

## What is worth flagging

In rough order of how often it has been the actual defect here:

1. **A change to one of a pair of twin functions**, with no statement about the other, or a guard
   placed in the copy the executing path does not call.
2. **A new output key that is not registered where registration is required.** Two places are
   mandatory: `config.all_observables`, or the writer never sees the key, and
   `interface/samplers.py: get_empty_result()`, or failed scan points produce rows that do not
   line up with successful ones. `interface/output_schema.py` is **not** mandatory:
   `_build_column_order()` appends every configured derived key absent from
   `DERIVED_PREFERRED_ORDER`, and then any remaining row key, so a key without an entry there is
   still written, just at the end. Add one only to place a column deliberately. Seventeen of the
   forty-six observables have no entry, including every `WARNING:` and `DIAG:` key, so do not
   report a missing one as a defect.
3. **Error and exit paths that skip bookkeeping the success path does**: a cache entry removed and
   not restored, a diagnostic carried on an exception and never recorded, a runtime recorded on
   one branch only.
4. **`except Exception` swallowing the run's own timeout.** `errors.Timeout` subclasses
   `Exception` and is raised from a signal handler, so it can fire anywhere. It must be re-raised;
   only genuine solver failures may be absorbed.
5. **Reusing a cached action.** `calcAction(pot, T, ..., outdict)` returns `outdict[T]["action"]`
   when present, so any recomputation at higher precision silently returns the old value unless
   the entry is removed first.
6. **A tolerance that loosens a stricter setting**, or a cap sized for a different stencil. Taking
   `min` is right; assignment is not.
7. **Counters that report what was planned rather than what happened**, and pairs of booleans that
   can be read as contradicting each other.
8. **Claims in prose that the code does not support.** Docstrings, `CHANGELOG.md` entries and
   numbers quoted in the pull request body are part of the change. A number measured before other
   work landed may be stale.

## Reviewing the tests

A test that passes is not evidence until it has been shown to fail without the fix. Flag a test
that cannot fail:

- it never reaches the new code path, because an earlier branch returns first;
- its synthetic perturbation is smaller than the threshold it is meant to cross;
- it injects a failure before the state under test is reached;
- its fixture is so minimal that a guard short-circuits (a bare `object()` where a potential is
  expected, so the real code raises and the assertion tests the fallback instead).

## Conventions

- Commit messages are one line, in the imperative, with no trailers.
- Pull request bodies are self-contained: no local paths, no references to unpushed branches.
- Numbers in a pull request body are measurements. If one cannot be reproduced it should be
  removed rather than softened.
- The full test suite must pass. Failures in `tests/test_release_smoke.py` usually mean the `tl`
  console script was not on `PATH`: its fallback invokes `transitionlistener.interface.cli` as a
  module, and that file has no `__main__` guard, so the subprocess does nothing and the fixture
  reports a missing output table. That is an environment problem, not a defect in the change.

## Please do not

- Re-report a finding that a later commit on the branch already fixes; check the current head.
- Ask for the two solvers to be unified as part of an unrelated change.
- Ask for a wall clock or other diagnostic to be moved in beside the physics observables.
