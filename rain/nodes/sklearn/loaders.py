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

from sklearn.datasets import load_iris

import pandas

from rain.core.base import Tags, LibTag, TypeTag, InputNode


class IrisDatasetLoader(InputNode):
    """Loads the iris dataset as a pandas DataFrame using the 'sklearn.datasets.load_iris'.

    Output
    ------
    dataset : pandas.DataFrame
        The iris dataset.
    target : pandas.DataFrame
        If separate_target is enabled then it will contain the target labels for the iris dataset.

    Parameters
    ----------
    node_id : str
        Id of the node.
    separate_target : bool, default=False
        Whether to get the target labels in the separated output 'target'.
    """

    _output_vars = {"dataset": pandas.DataFrame, "target": pandas.DataFrame}

    def __init__(self, node_id: str, separate_target: bool = False):
        self._separate_target = separate_target
        super(IrisDatasetLoader, self).__init__(node_id)

    def execute(self):
        if self._separate_target:
            self.dataset, self.target = load_iris(return_X_y=True, as_frame=True)
        else:
            self.dataset = load_iris(as_frame=True).data

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.SKLEARN, TypeTag.INPUT)
