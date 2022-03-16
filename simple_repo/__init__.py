"""Top-level package for SIMPLE Repository."""

__author__ = """Riccardo Coltrinari"""
__email__ = "riccardo.coltrinari@studenti.unicam.it"
__version__ = "0.1.0"

from . import _version

__version__ = _version.get_versions()["version"]

hard_dependencies = ["networkx", "numpy", "pandas"]
missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError:
        missing_dependencies.append(dependency)

if missing_dependencies:
    raise ImportError(f"Missing required dependencies:{','.join(missing_dependencies)}")

from simple_repo.core.base import *  # noqa: E402
from simple_repo.nodes import *  # noqa: E402
