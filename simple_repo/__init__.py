"""Top-level package for SIMPLE Repository."""

__author__ = """Riccardo Coltrinari"""
__email__ = "riccardo.coltrinari@studenti.unicam.it"
__version__ = "0.1.0"

from . import _version

__version__ = _version.get_versions()["version"]

from simple_repo.simple_io import *
from simple_repo.simple_pandas import *
from simple_repo.simple_sklearn import *
from simple_repo.simple_spark import *
from simple_repo.base import DataFlow
from simple_repo.custom import *
