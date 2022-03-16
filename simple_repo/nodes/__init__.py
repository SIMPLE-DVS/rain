from simple_repo.nodes.custom import *
from simple_repo.nodes.pandas import *


# Modular import of nodes: it is never correct to pass the exceptions
# but it is the only way I found to have both the modular import
# and the code completion by IDEs.
try:
    from simple_repo.nodes.sklearn import *
except ModuleNotFoundError or ImportError:
    pass

try:
    from simple_repo.nodes.mongodb import *
except ModuleNotFoundError or ImportError:
    pass

try:
    from simple_repo.nodes.spark import *
except ModuleNotFoundError or ImportError:
    pass

# The following code would be better for dynamical imports
# but the code completion does not work in this way.
# soft_dependencies = {
#     "pymongo": "simple_repo.mongodb",
#     "pyspark": "simple_repo.spark",
#     "sklearn": "simple_repo.sklearn"
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
