from pyspark.ml import PipelineModel

from simple_repo.simple_spark.spark_node import SparkParameter, SparkNode, Transformer


class SaveModel(SparkNode):

    _input = {
        "model": PipelineModel
    }

    _attr = {
        "path": SparkParameter("path", str, is_required=True)
    }

    def __init__(self, spark, **kwargs):
        super(SaveModel, self).__init__(spark, **kwargs)

    def execute(self):
        self.model.write().overwrite().save(**self._get_attr_as_dict())


class SaveDataset(Transformer):

    _attr = {
        "path": SparkParameter("path_or_buf", str, is_required=True),
        "index": SparkParameter("index", bool)
    }

    def __init__(self, spark, **kwargs):
        super(SaveDataset, self).__init__(spark, **kwargs)

    def execute(self):
        self.dataset.toPandas().to_csv(**self._get_attr_as_dict())
