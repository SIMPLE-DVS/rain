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

import pickle

from rain import OutputNode, InputNode, Tags, LibTag, TypeTag
from rain.core.parameter import Parameters, KeyValueParameter


class PickleModelWriter(OutputNode):
    """Node that stores a given object, for instance a trained model, in pickle format.

    Input
    -----
    model : pickle
        The object/model to store.

    Parameters
    ----------
    node_id : str
        Id of the node.
    path : str
        The path/filename where to store the object/model.
    """

    _input_vars = {"model": "pickle"}

    def __init__(self, node_id: str, path: str):
        super(PickleModelWriter, self).__init__(node_id)

        self.parameters = Parameters(
            path=KeyValueParameter("path", str, path)
        )

    def execute(self):
        pickle.dump(self.model, open(self.parameters.path.value, "wb"))

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.PANDAS, TypeTag.OUTPUT)


class PickleModelLoader(InputNode):
    """Node that loads a given object, for instance a trained model, stored in pickle format.

    Output
    ------
    model : pickle
        The loaded object in pickle format.

    Parameters
    ----------
    node_id : str
        Id of the node.
    path : str
        The path of the stored object/model.
    """

    _output_vars = {"model": "pickle"}

    def __init__(self, node_id: str, path: str):
        super(PickleModelLoader, self).__init__(node_id)

        self.parameters = Parameters(
            path=KeyValueParameter("path", str, path),
        )

    def execute(self):
        self.model = pickle.load(open(self.parameters.path.value, "rb"))

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.PANDAS, TypeTag.INPUT)
