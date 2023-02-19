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

from abc import abstractmethod

import pandas as pd

from rain.core.base import ComputationalNode, TypeTag, LibTag, Tags


class PandasTransformer(ComputationalNode):
    """Parent class for all the nodes that take a dataset as input, apply a transformation and expose the transformed dataset as output.

    Parameters
    ----------
    node_id : str
        Unique identifier of the node in the DataFlow.
    """

    _input_vars = {"dataset": pd.DataFrame}
    _output_vars = {"dataset": pd.DataFrame}

    def __init__(self, node_id: str):
        super(PandasTransformer, self).__init__(node_id)

    @abstractmethod
    def execute(self):
        pass  # pragma: no cover

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.PANDAS, TypeTag.TRANSFORMER)


class PandasNode(ComputationalNode):
    """
    Node that perform some transformation using the Pandas library without input/output constraints.

    Parameters
    ----------
    node_id : str
        Unique identifier of the node in the DataFlow.
    """

    def __init__(self, node_id: str):
        super(PandasNode, self).__init__(node_id)

    @abstractmethod
    def execute(self):
        pass  # pragma: no cover

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.PANDAS, TypeTag.TRANSFORMER)
