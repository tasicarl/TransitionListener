.. _faq:

===========================
Frequently Asked Questions
===========================

How can I ask a question not covered here?
------------------------------------------

See `Contributing <contributing.rst>`_ for how to report bugs and ask questions.

Opening a github issue is preferred, because then other people can find the question and answer.

How do I suppress the output?
-----------------------------

To suppress output, just call TransitionListener without the verbose flag

.. code-block:: console
  
  $ tl -c config/example_point.yaml

instead of 

.. code-block:: console
  
  $ tl -c config/example_point.yaml -v 

How should I cite TransitionListener?
-------------------------------------
Please cite the following papers if you use TransitionListener in your work:

.. code-block:: bibtex

  @article{Matuszak:2026xsz,
    author = "Matuszak, Jonas and Tasillo, Carlo",
    title = "{TransitionListener v2.0 -- Robust gravitational wave predictions for cosmological phase transitions}",
    eprint = "2605.15259",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    month = "5",
    year = "2026"
  }

  @article{Ertas:2021xeh,
      author = "Ertas, Fatih and Kahlhoefer, Felix and Tasillo, Carlo",
      title = "{Turn up the volume: listening to phase transitions in hot dark sectors}",
      eprint = "2109.06208",
      archivePrefix = "arXiv",
      primaryClass = "astro-ph.CO",
      reportNumber = "TTK-21-36, DESY-22-014",
      doi = "10.1088/1475-7516/2022/02/014",
      journal = "JCAP",
      volume = "02",
      number = "02",
      pages = "014",
      year = "2022"
  }
