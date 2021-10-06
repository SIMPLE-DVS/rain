from abc import abstractmethod

from pyspark.sql import DataFrame
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

from simple_repo.base import SimpleNode


class SparkNode(SimpleNode):
    _input_vars = {"spark": SparkSession}

    def __init__(self, spark):
        super(SparkNode, self).__init__()
        self.spark = spark

    @abstractmethod
    def execute(self):
        pass


class Transformer(SparkNode):
    _output_vars = {"dataset": DataFrame}

    def __init__(self, spark):
        self._input_vars["dataset"] = DataFrame
        super(Transformer, self).__init__(spark)

    @abstractmethod
    def execute(self):
        pass


class Estimator(SparkNode):
    _output_vars = {"model": PipelineModel}

    def __init__(self, spark):
        self._input_vars["dataset"] = DataFrame
        super(Estimator, self).__init__(spark)

    @abstractmethod
    def execute(self):
        pass
