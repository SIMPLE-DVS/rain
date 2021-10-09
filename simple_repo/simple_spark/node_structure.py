from abc import abstractmethod

from pyspark.sql import DataFrame
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

from simple_repo.base import ComputationalNode, InputNode, OutputNode


class SparkNodeSession:
    """Mixin class to share the spark session among different kinds of spark nodes."""

    spark: SparkSession = SparkSession.builder.getOrCreate()


class SparkStageMixin:
    def __init__(self):
        self.computational_instance = None


class SparkNode(ComputationalNode, SparkNodeSession, SparkStageMixin):
    """Class representing a Spark ComputationalNode, it could be either a Transformer or Estimator."""

    _input_vars = {"dataset": DataFrame}

    def __init__(self):
        SparkStageMixin.__init__(self)
        super(SparkNode, self).__init__()

    @abstractmethod
    def execute(self):
        pass


class Transformer(SparkNode):
    """Class representing a Spark Transformer, it manipulates a given dataset and returns a modified version of it."""

    _output_vars = {"dataset": DataFrame}

    def __init__(self):
        super(Transformer, self).__init__()

    @abstractmethod
    def execute(self):
        pass


class Estimator(SparkNode):
    """Class representing a Spark Estimator, it takes a dataset and returns a trained model."""

    _output_vars = {"model": PipelineModel}

    def __init__(self):
        super(Estimator, self).__init__()

    @abstractmethod
    def execute(self):
        pass


class SparkInputNode(InputNode, SparkNodeSession):
    """Class representing a Spark InputNode, it loads and returns an object/file."""

    def __init__(self):
        pass

    @abstractmethod
    def execute(self):
        pass


class SparkOutputNode(OutputNode, SparkNodeSession):
    """Class representing a Spark OutputNode, it save a given object/file."""

    def __init__(self):
        pass

    @abstractmethod
    def execute(self):
        pass
