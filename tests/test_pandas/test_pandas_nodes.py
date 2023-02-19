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

import pytest

from rain.nodes.pandas.node_structure import PandasTransformer
import rain.nodes.pandas as sp
import rain.nodes.pandas.pandas_io as pio

pandas_nodes = [
    sp.PandasColumnsFiltering,
    sp.PandasPivot,
    sp.PandasRenameColumn,
    pio.PandasCSVLoader,
    pio.PandasCSVWriter,
    sp.PandasFilterRows,
    sp.PandasAddColumn,
    sp.PandasDropNan,
    # sp.PandasSelectRows,
    # sp.PandasReplaceColumn
]


@pytest.mark.parametrize("class_or_obj", pandas_nodes)
def test_pandas_methods(class_or_obj):
    """Checks whether the spark node has no method attributes."""
    assert (
        issubclass(class_or_obj, PandasTransformer)
        or issubclass(class_or_obj, pio.PandasInputNode)
        or issubclass(class_or_obj, pio.PandasOutputNode)
    ) and not hasattr(class_or_obj, "_methods")
