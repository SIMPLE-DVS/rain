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

try:
    from rain.nodes.pysad import *
except ModuleNotFoundError or ImportError:
    pass
