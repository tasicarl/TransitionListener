# Notes for coding agents

The review brief for this repository lives in `.github/copilot-instructions.md`. Read it before
reviewing or changing anything. It is written for automated code review but applies to any agent
working here: it describes the parallel adaptive / fixed-step solver modules, which duplications
are deliberate, where a new output key has to be registered, and the defect classes that recur.

Two points from it are worth repeating, because they are the ones most often missed:

- **Many functions exist twice**, once in the adaptive module and once in its `_fixedstep` twin.
  Changing one copy does not reach the other, and a guard added to the copy the executing path
  does not call is dead code. Before finishing any change, list the symbols it touches and grep
  the source tree for each of them.
- **A passing test is not evidence.** Revert the change and confirm the test fails before
  claiming it covers anything.

Run the test suite with the source tree on the path, not the editable install:

    PYTHONPATH=$PWD/src python -m pytest -q -p no:cacheprovider tests
