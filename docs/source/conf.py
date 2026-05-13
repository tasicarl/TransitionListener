# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import warnings
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))  # Ensure autodoc finds the package

from sphinx.deprecation import RemovedInSphinx11Warning

warnings.filterwarnings("ignore", category=RemovedInSphinx11Warning)


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'TransitionListener'
copyright = '2025, Carlo Tasillo, Jonas Matuszak'
author = 'Carlo Tasillo, Jonas Matuszak'
release = '2.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser', # Markdown support
    'sphinx.ext.autodoc', # pull docstrings
    'sphinx.ext.napoleon', # Google/NumPy style docstrings
    'sphinx.ext.intersphinx',
    "sphinx.ext.githubpages",
    "sphinx_autodoc_typehints",
    "sphinx.ext.todo",
    "matplotlib.sphinxext.plot_directive",
]

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]


templates_path = ['_templates']
exclude_patterns = []
suppress_warnings = ["ref.python"]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_title = "TransitionListener"
html_short_title = "TransitionListener"
html_logo = str(project_root / "src" / "transitionlistener" / "logo" / "TL-logo_small.png")
html_show_sourcelink = False
html_theme_options = {
    "sidebar_hide_name": True,
}

pygments_style = "sphinx"       # enable syntax highlighting

# Napoleon settings for Google/NumPy-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Additional extension settings
todo_include_todos = True
autodoc_default_options = {
    "noindex": True,
}
