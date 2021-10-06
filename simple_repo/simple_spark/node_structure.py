from abc import abstractmethod

from pyspark.sql import DataFrame
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

from simple_repo.base import ComputationalNode, InputNode, OutputNode


class SparkNode:
    spark: SparkSession = None


class Transformer(ComputationalNode, SparkNode):
    _input_vars = {"dataset": DataFrame}

    _output_vars = {"dataset": DataFrame}

    def __init__(self):
        super(Transformer, self).__init__()

    @abstractmethod
    def execute(self):
        pass


class Estimator(ComputationalNode, SparkNode):
    _input_vars = {"dataset": DataFrame}

    _output_vars = {"model": PipelineModel}

    def __init__(self):
        super(Estimator, self).__init__()

    @abstractmethod
    def execute(self):
        pass


class SparkInputNode(InputNode, SparkNode):
    def __init__(self):
        pass

    @abstractmethod
    def execute(self):
        pass


class SparkOutputNode(OutputNode, SparkNode):
    def __init__(self):
        pass

    @abstractmethod
    def execute(self):
        pass
