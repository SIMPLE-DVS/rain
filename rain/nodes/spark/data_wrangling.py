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

from typing import List

from pyspark.sql import DataFrame

from rain.core.parameter import KeyValueParameter, Parameters
from rain.nodes.spark.node_structure import Transformer


class SparkColumnSelector(Transformer):
    """SparkColumnSelector manages filtering of rows, columns and values for a Spark DataFrame.

    Input
    -----
    dataset : DataFrame
        A Spark DataFrame.

    Output
    ------
    dataset : DataFrame
        A Spark DataFrame.

    Parameters
    ----------
    node_id : str
        Id of the node
    column_list : List[str]
        List of columns to select from the dataset
    filter_list : List[str]
        List of conditions used to filter the rows of the dataset
    """

    def __init__(
        self, node_id: str, column_list: List[str], filter_list: List[str] = []
    ):
        super(SparkColumnSelector, self).__init__(node_id)
        self.parameters = Parameters(
            column_list=KeyValueParameter("column_list", List[str], column_list),
            filter_list=KeyValueParameter("filter_list", List[str], filter_list),
        )

    def execute(self):
        self.dataset = self.dataset.select(
            self.parameters.get_dict().get("column_list")
        )
        for c in self.parameters.get_dict().get("filter_list"):
            self.dataset = self.dataset.filter(c)


class SparkSplitDataset(Transformer):
    """Splits a Spark DataFrame in two DataFrames, train and test.

    Input
    -----
    dataset : DataFrame
        A Spark DataFrame.

    Output
    ------
    train_dataset : DataFrame
        A Spark DataFrame used for the training phase.
    test_dataset : DataFrame
        A Spark DataFrame used for the test phase.

    Parameters
    ----------
    node_id : str
        Id of the node.
    train : float
        Percentage of the dataset to split into a train dataset.
    test : float
        Percentage of the dataset to split into a test dataset.
    """

    _output_vars = {"train_dataset": DataFrame, "test_dataset": DataFrame}

    def __init__(self, node_id: str, train: float, test: float):
        super(SparkSplitDataset, self).__init__(node_id)
        self.parameters = Parameters(
            train=KeyValueParameter("train", float, train),
            test=KeyValueParameter("test", float, test),
        )

    def execute(self):
        values = list(self.parameters.get_dict().values())
        self.train_dataset, self.test_dataset = self.dataset.randomSplit(values)
