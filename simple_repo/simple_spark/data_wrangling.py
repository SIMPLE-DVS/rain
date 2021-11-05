from typing import List

from pyspark.sql import DataFrame

from simple_repo.parameter import KeyValueParameter, Parameters
from simple_repo.simple_spark.node_structure import Transformer


class SparkColumnSelector(Transformer):
    """SparkColumnSelector manages filtering of rows, columns and values
    for a Spark DataFrame.

    Parameters
    ----------
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

    Parameters
    ----------
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
