from pyspark.sql import DataFrame

from simple_repo.simple_spark.spark_node import Transformer, SparkParameterList, SparkParameter


class SparkColumnSelector(Transformer):
    _attr = {
        "features": SparkParameterList(col=True, value=False)
    }

    def __init__(self, spark, **kwargs):
        super(SparkColumnSelector, self).__init__(spark, **kwargs)

    def execute(self):
        columns = [c["col"] for c in self._attr.get("features").parameters]
        self.dataset = self.dataset.select(columns)
        conditions = [c["value"] for c in self._attr.get("features").parameters if c["value"]]
        for c in conditions:
            self.dataset = self.dataset.filter(c)
        self.dataset.show()


class SparkSplitDataset(Transformer):
    _attr = {
        "train": SparkParameter("train", float, is_required=True),
        "test": SparkParameter("test", float, is_required=True)
    }

    _output = {
        "train_dataset": DataFrame,
        "test_dataset": DataFrame
    }

    def __init__(self, spark, **kwargs):
        super(SparkSplitDataset, self).__init__(spark, **kwargs)

    def execute(self):
        values = list(self._get_attr_as_dict().values())
        self.train_dataset, self.test_dataset = self.dataset.randomSplit(values)
        self.train_dataset.show()
        self.test_dataset.show()


