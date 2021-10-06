from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame

from simple_repo.parameter import KeyValueParameter, Parameters
from simple_repo.simple_spark.node_structure import SparkNode


class SaveModel(SparkNode):
    """ Save a trained PipelineModel

    Parameters
    ----------
    path: str
        String representing the path where to save the model

    """

    _input_vars = {"model": PipelineModel}

    def __init__(self, spark, path: str):
        self.parameters = Parameters(path=KeyValueParameter("path", str, path))
        super(SaveModel, self).__init__(spark)

    def execute(self):
        self.model.write().overwrite().save(**self.parameters.get_dict())


class SaveDataset(SparkNode):
    """ Save a Spark Dataframe in a .csv format

    Parameters
    ----------
    path: str
        String representing the path where to save the dataset

    index: bool = True
        String representing the path where to save the dataset
    """

    _input_vars = {"dataset": DataFrame}

    def __init__(self, spark, path: str, index: bool = True):
        self.parameters = Parameters(
            path=KeyValueParameter("path_or_buf", str, path),
            index=KeyValueParameter("index", bool, index)
        )
        super(SaveDataset, self).__init__(spark)

    def execute(self):
        self.dataset.toPandas().to_csv(**self.parameters.get_dict())
