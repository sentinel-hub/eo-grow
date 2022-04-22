# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import shutil

# -- Project information -----------------------------------------------------

# General information about the project.
project = "EO Grow"
project_copyright = "2022, Sentinel Hub"
author = "Sinergise EO research team"
doc_title = "eogrow Documentation"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The release is read from __init__ file and version is shortened release string.
for line in open(os.path.join(os.path.dirname(__file__), "../../eogrow/__init__.py")):
    if line.find("__version__") >= 0:
        release = line.split("=")[1].strip()
        release = release.strip('"').strip("'")
version = release.rsplit(".", 1)[0]

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "nbsphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "m2r2",
]

# Both the class’ and the __init__ method’s docstring are concatenated and inserted.
autoclass_content = "both"

# Content is in the same order as in module
autodoc_member_order = "bysource"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# General information about the project.


# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["**.ipynb_checkpoints"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# Mock imports that won't and don't have to be installed in ReadTheDocs environment
autodoc_mock_imports = ["ray"]

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# html_logo = "./sentinel-hub-by_sinergise-dark_background.png"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {
#     "rightsidebar": "true",
#     "relbarbgcolor": "black"}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# This is required for the alabaster theme
# refs: http://alabaster.readthedocs.io/en/latest/installation.html#sidebars
# html_sidebars = {
#    '**': [
#        'about.html',
#        'navigation.html',
#        'relations.html',  # needs 'show_related': True theme option to display
#        'searchbox.html',
#        'donate.html',
#    ]
# }

# analytics
# html_js_files = [("https://cdn.usefathom.com/script.js", {"data-site": "BILSIGFB", "defer": "defer"})]


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "eogrow_doc"
# show/hide links for source
html_show_sourcelink = False

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "eo-grow.tex", doc_title, author, "manual"),
]

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "eo-grow", doc_title, [author], 1)]

# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, "eo-grow", doc_title, author, "eo-grow", "One line description of project.", "Miscellaneous"),
]

# -- Options for Epub output ----------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = project_copyright

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ["search.html"]

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {"https://docs.python.org/3.8/": None}

# copy examples
# try:
#     shutil.copytree("../../examples", "./examples")
# except FileExistsError:
#     pass


MARKDOWNS_FOLDER = "./markdowns"
shutil.rmtree(MARKDOWNS_FOLDER, ignore_errors=True)
os.mkdir(MARKDOWNS_FOLDER)

def process_readme():
    """Function which will process README.md file and create INTRO.md"""
    with open("../../README.md", "r") as file:
        readme = file.read()

    readme = readme.replace("[`", "[").replace("`]", "]").replace("docs/source/", "")
    readme = readme.replace("**`", "**").replace("`**", "**")

    chapters = [[]]
    for line in readme.split("\n"):
        if line.strip().startswith("## "):
            chapters.append([])
        if line.startswith("<img"):
            line = "<p></p>"

        chapters[-1].append(line)

    chapters = ["\n".join(chapter) for chapter in chapters]

    intro = "\n".join(
        [
            chapter
            for chapter in chapters
            if not (chapter.startswith("## Install") or chapter.startswith("## Documentation"))
        ]
    )

    with open(os.path.join(MARKDOWNS_FOLDER, "INTRO.md"), "w") as file:
        file.write(intro)


process_readme()
