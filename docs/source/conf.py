# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
from os.path import dirname



SOURCE = os.path.dirname(os.path.realpath(__file__))



sys.path.insert(0, SOURCE)

project = 'TransOPT: Transfer Optimization System for Bayesian Optimization Using Transfer Learning'
copyright = '2024, Peili Mao'
author = 'Peili Mao'
release = '0.1.0'




# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions =[
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    'sphinxcontrib.bibtex',
    
    'sphinx_togglebutton',
    
    'sphinx.ext.mathjax',
    'sphinx.ext.autosummary',
    # 'numpydoc',
    # 'nbsphinx',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    # 'matplotlib.sphinxext.plot_directive',
    ]

templates_path = ['_templates']
exclude_patterns = []

bibtex_bibfiles = ['usage/TOS.bib']

html_logo = "_static//figures/transopt_logo.jpg"
# html_favicon = '_static/favicon.ico'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]



master_doc = 'index'