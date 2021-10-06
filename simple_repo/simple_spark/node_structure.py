from abc import abstractmethod

from pyspark.sql import DataFrame
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

from simple_repo.base import ComputationalNode, InputNode, OutputNode


class SparkNodeSession:
    """Mixin class to share the spark session among different kinds of spark nodes."""

    spark: SparkSession = None


class Transformer(ComputationalNode, SparkNodeSession):
    _input_vars = {"dataset": DataFrame}

    _output_vars = {"dataset": DataFrame}

    def __init__(self):
        super(Transformer, self).__init__()

    @abstractmethod
    def execute(self):
        pass


class Estimator(ComputationalNode, SparkNodeSession):
    _input_vars = {"dataset": DataFrame}

    _output_vars = {"model": PipelineModel}

    def __init__(self):
        super(Estimator, self).__init__()

    @abstractmethod
    def execute(self):
        pass


class SparkInputNode(InputNode, SparkNodeSession):
    def __init__(self):
        pass

    @abstractmethod
    def execute(self):
        pass


class SparkOutputNode(OutputNode, SparkNodeSession):
    def __init__(self):
        pass

    @abstractmethod
    def execute(self):
        pass
