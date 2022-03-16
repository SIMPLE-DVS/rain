from abc import abstractmethod

from pyspark.sql import DataFrame
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

from rain.core.base import (
    ComputationalNode,
    InputNode,
    OutputNode,
    TypeTag,
    LibTag,
    Tags,
)


class SparkNodeSession:
    """Mixin class to share the spark session among different kinds of spark nodes."""

    spark: SparkSession = None


class SparkStageMixin:
    def __init__(self):
        self.computational_instance = None


class SparkNode(ComputationalNode, SparkNodeSession, SparkStageMixin):
    """Class representing a Spark ComputationalNode, it could be either a Transformer or Estimator."""

    _input_vars = {"dataset": DataFrame}

    def __init__(self, node_id):
        SparkStageMixin.__init__(self)
        super(SparkNode, self).__init__(node_id)

    @abstractmethod
    def execute(self):
        pass


class Transformer(SparkNode):
    """Class representing a Spark Transformer, it manipulates a given dataset and returns a modified version of it."""

    _output_vars = {"dataset": DataFrame}

    def __init__(self, node_id: str):
        super(Transformer, self).__init__(node_id)

    @abstractmethod
    def execute(self):
        pass

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.SPARK, TypeTag.TRANSFORMER)


class Estimator(SparkNode):
    """Class representing a Spark Estimator, it takes a dataset and returns a trained model."""

    _output_vars = {"model": PipelineModel}

    def __init__(self, node_id: str):
        super(Estimator, self).__init__(node_id)

    @abstractmethod
    def execute(self):
        pass

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.SPARK, TypeTag.ESTIMATOR)


class SparkInputNode(InputNode, SparkNodeSession):
    """Class representing a Spark InputNode, it loads and returns an object/file."""

    @abstractmethod
    def execute(self):
        pass

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.SPARK, TypeTag.INPUT)


class SparkOutputNode(OutputNode, SparkNodeSession):
    """Class representing a Spark OutputNode, it save a given object/file."""

    @abstractmethod
    def execute(self):
        pass

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.SPARK, TypeTag.OUTPUT)
