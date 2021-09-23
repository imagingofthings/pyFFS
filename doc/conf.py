# #############################################################################
# conf.py
# =======
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# Author : Eric BEZZAM [ebezzam@gmail.com]
# #############################################################################

import configparser
import datetime
import pathlib


autodoc_mock_imports = ["numpy", "scipy", "cupy", "cupyx"]


def setup_config() -> configparser.ConfigParser:
    """
    Load information contained in `setup.cfg`.
    """
    sphinx_src_dir = pathlib.Path(__file__).parent
    setup_path = sphinx_src_dir / ".." / "setup.cfg"
    setup_path = setup_path.resolve(strict=True)

    with setup_path.open(mode="r") as f:
        cfg = configparser.ConfigParser()
        cfg.read_file(f)
    return cfg


# -- Project information -----------------------------------------------------
cfg = setup_config()
project = cfg.get("metadata", "name")
copyright = f"{datetime.date.today().year}, Imaging of Things Group (ImoT)"
author = cfg.get("metadata", "author")

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
master_doc = "index"
exclude_patterns = []
pygments_style = "sphinx"
add_module_names = False

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {"navigation_depth": -1, "titles_only": True}

# -- Options for HTMLHelp output ---------------------------------------------
htmlhelp_basename = "pyffsdoc"

# -- Extension configuration -------------------------------------------------
# -- Options for autosummary extension ---------------------------------------
autosummary_generate = True

# -- Options for autodoc extension -------------------------------------------
autodoc_member_order = "bysource"
autodoc_default_flags = [
    "members",
    # 'inherited-members',
    "show-inheritance",
]
autodoc_inherit_docstrings = True

# -- Options for intersphinx extension ---------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "NumPy [latest]": ("https://docs.scipy.org/doc/numpy/", None),
    "SciPy [latest]": ("https://docs.scipy.org/doc/scipy/reference", None),
}

# -- Options for napoleon extension ------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# -- Options for todo extension ----------------------------------------------
todo_include_todos = True
