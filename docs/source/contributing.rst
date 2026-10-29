.. highlight:: shell

============
Contributing
============

Contributions to TransitionListener are very welcome! Every little bit helps,
and credit will always be given. You can contribute in many ways:

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/tasicarl/TransitionListener/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

TransitionListener could always use more documentation, whether as part of the
official TransitionListener docs, in docstrings, or even on the web in blog posts,
articles, and such.

Notebooks demonstrating how to use TransitionListener are also appreciated.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at
https://github.com/tasicarl/TransitionListener/issues.

Conventions in the source
~~~~~~~~~~~~~~~~~~~~~~~~~

One convention is easy to undo by accident. A run's time limit is delivered as
:class:`~transitionlistener.errors.Timeout`, raised from a POSIX alarm handler, so it can
arrive on any bytecode, including inside a routine whose own failures are converted into a
stage-specific error. Every broad ``except Exception`` that performs such a conversion is
therefore preceded by::

    except errors.Timeout:
        raise

Those two lines look redundant and are not. Without them the time limit is caught by the
handler below and reported as a physics verdict, so a run that never finished comes out as a
finished one. There are 42 of them. Keep the guard whenever you add a broad handler, and do
not merge it into the handler below.
