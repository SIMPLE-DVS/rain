from simple_repo.simple_spark.spark_node import SparkParameter, Transformer


class SparkCSVLoader(Transformer):

    _attr = {
        "path": SparkParameter("path", str, is_required=True),
        "header": SparkParameter("header", bool),
        "schema": SparkParameter("inferSchema", bool)
    }

    def __init__(self, spark, **kwargs):
        super(SparkCSVLoader, self).__init__(spark, **kwargs)

    def execute(self):
        self.dataset = self.spark.read.csv(**self._get_attr_as_dict())
