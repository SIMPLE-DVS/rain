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

from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame

from rain.core.parameter import KeyValueParameter, Parameters
from rain.nodes.spark.node_structure import SparkOutputNode


class SparkSaveModel(SparkOutputNode):
    """Save a trained PipelineModel

    Input
    -----
    model : PipelineModel
        The Spark PipelineModel to be saved.

    Parameters
    ----------
    node_id : str
        Id of the node.
    path : str
        String representing the path where to save the model.

    """

    _input_vars = {"model": PipelineModel}

    def __init__(self, node_id: str, path: str):
        self.parameters = Parameters(path=KeyValueParameter("path", str, path))
        super(SparkSaveModel, self).__init__(node_id)

    def execute(self):
        self.model.write().overwrite().save(**self.parameters.get_dict())


class SparkSaveDataset(SparkOutputNode):
    """Save a Spark Dataframe in a .csv format

    Input
    -----
    dataset : DataFrame
        The Spark Dataframe to be saved.

    Parameters
    ----------
    node_id : str
        Id of the node.
    path : str
        String representing the path where to save the dataset

    index : bool, default True
        String representing the path where to save the dataset
    """

    _input_vars = {"dataset": DataFrame}

    def __init__(self, node_id: str, path: str, index: bool = True):
        self.parameters = Parameters(
            path=KeyValueParameter("path_or_buf", str, path),
            index=KeyValueParameter("index", bool, index),
        )
        super(SparkSaveDataset, self).__init__(node_id)

    def execute(self):
        self.dataset.toPandas().to_csv(**self.parameters.get_dict())
