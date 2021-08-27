from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame

from simple_repo.parameter import KeyValueParameter
from simple_repo.simple_spark.node_structure import SparkNode


class SparkCSVLoader(SparkNode):
    _parameters = {
        "path": KeyValueParameter("path", str, is_mandatory=True),
        "header": KeyValueParameter("header", bool),
        "schema": KeyValueParameter("inferSchema", bool)
    }

    _output_vars = {
        "dataset": DataFrame
    }

    def __init__(self, spark, **kwargs):
        super(SparkCSVLoader, self).__init__(spark, **kwargs)

    def execute(self):
        self.dataset = self.spark.read.csv(**self._get_params_as_dict())


class SparkModelLoader(SparkNode):
    _parameters = {
        "path": KeyValueParameter("path", str, is_mandatory=True)
    }

    _output_vars = {
        "model": PipelineModel
    }

    def __init__(self, spark, **kwargs):
        super(SparkModelLoader, self).__init__(spark, **kwargs)

    def execute(self):
        self.model = PipelineModel.load(self._parameters.get("path").value)
