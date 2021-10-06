from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame

from simple_repo.parameter import KeyValueParameter, Parameters
from simple_repo.simple_spark.node_structure import SparkNode


class SparkCSVLoader(SparkNode):
    """Loads a CSV file as a Spark DataFrame.

    Parameters
    ----------
    path : str
        Path of the csv file.
    header : bool
        uses the first line as names of columns. If None is set, it uses the
        default value, ``false``.
    schema : bool
        infers the input schema automatically from data. It requires one extra
        pass over the data. If None is set, it uses the default value, ``false``.

    """

    _output_vars = {"dataset": DataFrame}

    def __init__(self, path: str, header: bool = None, schema: bool = None):
        self.parameters = Parameters(
            path=KeyValueParameter("path", str, path),
            header=KeyValueParameter("header", bool, header),
            schema=KeyValueParameter("inferSchema", bool, schema),
        )
        super(SparkCSVLoader, self).__init__()

    def execute(self):
        self.dataset = self.spark.read.csv(**self.parameters.get_dict())


class SparkModelLoader(SparkNode):
    """Loads a file as a Spark Model.

    Parameters
    ----------
    path : str
        Path of the csv file.
    """

    _output_vars = {"model": PipelineModel}

    def __init__(self, path: str):
        self.parameters = Parameters(path=KeyValueParameter("path", str, path))
        super(SparkModelLoader, self).__init__()

    def execute(self):
        self.model = PipelineModel.load(self.parameters.path.value)
