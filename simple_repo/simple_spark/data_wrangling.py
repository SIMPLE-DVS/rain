from pyspark.sql import DataFrame

from simple_repo.parameter import StructuredParameterList, KeyValueParameter, Parameters
from simple_repo.simple_spark.node_structure import Transformer


class SparkColumnSelector(Transformer):
    """SparkColumnSelector manages filtering of rows, columns and values
    for a Spark DataFrame.

    Parameters
    ----------
    features : list[dict]
        Every dictionary in the list must be of the form:
            {
                col: str (Mandatory)
                value: str (Optional)
            }
    """

    def __init__(self, node_id: str, features: list):
        super(SparkColumnSelector, self).__init__(node_id)
        self.parameters = Parameters(
            features=StructuredParameterList(col=True, value=False)
        )
        self.parameters.features.add_all_parameters(features)

    def execute(self):
        columns = [c["col"] for c in self.parameters.features.parameters]
        self.dataset = self.dataset.select(columns)
        conditions = [
            c["value"] for c in self.parameters.features.parameters if "value" in c
        ]
        for c in conditions:
            self.dataset = self.dataset.filter(c)
        self.dataset.show()


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
        self.train_dataset.show()
        self.test_dataset.show()
