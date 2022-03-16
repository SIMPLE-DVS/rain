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

from simple_repo.base import DataFlow  # noqa: E402
from simple_repo.custom import *  # noqa: E402
from simple_repo.simple_pandas import *  # noqa: E402


# Modular import of nodes: it is never correct to pass the exceptions
# but it is the only way I found to have both the modular import
# and the code completion by IDEs.
try:
    from simple_repo.simple_sklearn import *
except ModuleNotFoundError or ImportError:
    pass

try:
    from simple_repo.mongodb import *
except ModuleNotFoundError or ImportError:
    pass

try:
    from simple_repo.simple_spark import *
except ModuleNotFoundError or ImportError:
    pass

# The following code would be better for dynamical imports
# but the code completion does not work in this way.
# soft_dependencies = {
#     "pymongo": "simple_repo.mongodb",
#     "pyspark": "simple_repo.simple_spark",
#     "sklearn": "simple_repo.simple_sklearn"
# }
#
# for dependency, plugin_module in soft_dependencies.items():
#     try:
#         __import__(dependency)
#     except ImportError as e:
#         continue
#
#     module = importlib.import_module(plugin_module)
#     for attribute_name in dir(module):
#         attribute = getattr(module, attribute_name)
#
#         if isclass(attribute) and issubclass(attribute, base.SimpleNode):
#             # Add the class to this package's variables
#             globals()[attribute_name] = attribute
