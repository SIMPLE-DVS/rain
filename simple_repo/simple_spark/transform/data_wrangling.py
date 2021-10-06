from pyspark.sql import DataFrame

from simple_repo.parameter import StructuredParameterList, KeyValueParameter
from simple_repo.simple_spark.node_structure import Transformer


class SparkColumnSelector(Transformer):
    _parameters = {"features": StructuredParameterList(col=True, value=False)}

    def __init__(self, spark, **kwargs):
        super(SparkColumnSelector, self).__init__(spark)

    def execute(self):
        columns = [c["col"] for c in self._parameters.get("features").parameters]
        self.dataset = self.dataset.select(columns)
        conditions = [
            c["value"]
            for c in self._parameters.get("features").parameters
            if "value" in c
        ]
        for c in conditions:
            self.dataset = self.dataset.filter(c)
        self.dataset.show()


class SparkSplitDataset(Transformer):
    _parameters = {
        "train": KeyValueParameter("train", float, is_mandatory=True),
        "test": KeyValueParameter("test", float, is_mandatory=True),
    }

    _output_vars = {"train_dataset": DataFrame, "test_dataset": DataFrame}

    def __init__(self, spark, **kwargs):
        super(SparkSplitDataset, self).__init__(spark)

    def execute(self):
        values = list(self._get_params_as_dict().values())
        self.train_dataset, self.test_dataset = self.dataset.randomSplit(values)
        self.train_dataset.show()
        self.test_dataset.show()
