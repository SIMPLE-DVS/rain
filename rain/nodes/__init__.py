"""
 Copyright (C) 2023 Università degli Studi di Camerino and Sigma S.p.A.
 Authors: Alessandro Antinori, Rosario Capparuccia, Riccardo Coltrinari, Flavio Corradini, Marco Piangerelli, Barbara Re, Marco Scarpetta

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU Affero General Public License as
 published by the Free Software Foundation, either version 3 of the
 License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Affero General Public License for more details.

 You should have received a copy of the GNU Affero General Public License
 along with this program.  If not, see <https://www.gnu.org/licenses/>.
 """

from rain.nodes.custom import *

# Modular import of nodes: it is never correct to pass the exceptions
# but it is the only way I found to have both the modular import
# and the code completion by IDEs.
try:
    from rain.nodes.pandas import *
except ModuleNotFoundError or ImportError:
    pass

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
