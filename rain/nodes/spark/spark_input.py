from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame

from rain.core.parameter import KeyValueParameter, Parameters
from rain.nodes.spark.node_structure import SparkInputNode


class SparkCSVLoader(SparkInputNode):
    """Loads a CSV file as a Spark DataFrame.

    Parameters
    ----------
    path : str
        Path of the csv file.
    header : bool, default False
        Uses the first line as names of columns.
    schema : bool, default False
        Infers the input schema automatically from data. It requires one extra
        pass over the data.
    """

    _output_vars = {"dataset": DataFrame}

    def __init__(
        self, node_id: str, path: str, header: bool = False, schema: bool = False
    ):
        super(SparkCSVLoader, self).__init__(node_id)
        self.parameters = Parameters(
            path=KeyValueParameter("path", str, path),
            header=KeyValueParameter("header", bool, header),
            schema=KeyValueParameter("inferSchema", bool, schema),
        )

    def execute(self):
        self.dataset = self.spark.read.csv(**self.parameters.get_dict())


class SparkModelLoader(SparkInputNode):
    """Loads a file as a Spark Model.

    Parameters
    ----------
    path : str
        Path of the csv file.
    """

    _output_vars = {"model": PipelineModel}

    def __init__(self, node_id: str, path: str):
        self.parameters = Parameters(path=KeyValueParameter("path", str, path))
        super(SparkModelLoader, self).__init__(node_id)

    def execute(self):
        self.model = PipelineModel.load(self.parameters.path.value)
