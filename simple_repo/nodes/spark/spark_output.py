from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame

from simple_repo.core.parameter import KeyValueParameter, Parameters
from simple_repo.nodes.spark.node_structure import SparkOutputNode


class SparkSaveModel(SparkOutputNode):
    """Save a trained PipelineModel

    Parameters
    ----------
    path : str
        String representing the path where to save the model

    """

    _input_vars = {"model": PipelineModel}

    def __init__(self, node_id: str, path: str):
        self.parameters = Parameters(path=KeyValueParameter("path", str, path))
        super(SparkSaveModel, self).__init__(node_id)

    def execute(self):
        self.model.write().overwrite().save(**self.parameters.get_dict())


class SparkSaveDataset(SparkOutputNode):
    """Save a Spark Dataframe in a .csv format

    Parameters
    ----------
    path : str
        String representing the path where to save the dataset

    index : bool, default True
        String representing the path where to save the dataset
    """

    _input_vars = {"dataset": DataFrame}

    def __init__(self, node_id: str, path: str, index: bool = True):
        self.parameters = Parameters(
            path=KeyValueParameter("path_or_buf", str, path),
            index=KeyValueParameter("index", bool, index),
        )
        super(SparkSaveDataset, self).__init__(node_id)

    def execute(self):
        self.dataset.toPandas().to_csv(**self.parameters.get_dict())
