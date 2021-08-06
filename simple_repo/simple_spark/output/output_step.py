from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame

from simple_repo.parameter import KeyValueParameter
from simple_repo.simple_spark.spark_node import SparkNode


class SaveModel(SparkNode):
    _input_vars = {
        "model": PipelineModel
    }

    _parameters = {
        "path": KeyValueParameter("path", str, is_mandatory=True)
    }

    def __init__(self, spark, **kwargs):
        super(SaveModel, self).__init__(spark, **kwargs)

    def execute(self):
        self.model.write().overwrite().save(**self._get_params_as_dict())


class SaveDataset(SparkNode):
    _input_vars = {
        "dataset": DataFrame
    }

    _attr = {
        "path": KeyValueParameter("path_or_buf", str, is_mandatory=True),
        "index": KeyValueParameter("index", bool)
    }

    def __init__(self, spark, **kwargs):
        super(SaveDataset, self).__init__(spark, **kwargs)

    def execute(self):
        self.dataset.toPandas().to_csv(**self._get_params_as_dict())
