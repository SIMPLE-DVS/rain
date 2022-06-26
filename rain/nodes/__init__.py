from rain.nodes.custom import *
from rain.nodes.pandas import *


# Modular import of nodes: it is never correct to pass the exceptions
# but it is the only way I found to have both the modular import
# and the code completion by IDEs.
try:
    from rain.nodes.sklearn import *
except ModuleNotFoundError or ImportError:
    pass

try:
    from rain.nodes.mongodb import *
except ModuleNotFoundError or ImportError:
    pass

try:
    from rain.nodes.spark import *
except ModuleNotFoundError or ImportError:
    pass

try:
    from rain.nodes.tpot import *
except ModuleNotFoundError or ImportError:
    pass

# The following code would be better for dynamical imports
# but the code completion does not work in this way.
# soft_dependencies = {
#     "pymongo": "rain.mongodb",
#     "pyspark": "rain.spark",
#     "sklearn": "rain.sklearn",
#     "tpot": "rain.tpot"
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
