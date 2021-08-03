from simple_repo.simple_spark.spark_node import Transformer, SparkParameterList


class SparkColumnSelector(Transformer):
    _attr = {
        "features": SparkParameterList(col=True, value=False)
    }

    def __init__(self, spark, **kwargs):
        super(SparkColumnSelector, self).__init__(spark, **kwargs)

    def execute(self):
        columns = [c["col"] for c in self._attr.get("features").parameters]
        self.dataset = self.dataset.select(columns)
