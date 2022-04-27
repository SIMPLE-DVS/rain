from rain.core.parameter import KeyValueParameter, Parameters
from rain.nodes.spark.node_structure import Transformer, Estimator
from pyspark.ml.classification import LogisticRegression as Lr
from pyspark.ml.feature import HashingTF as Htf, Tokenizer as Tk


class Tokenizer(Transformer):
    """Represent a Spark Tokenizer used to split text in individual term.

    Input
    -----
    dataset : DataFrame
        A Spark DataFrame.

    Output
    ------
    dataset : DataFrame
        The modified Spark DataFrame.

    Parameters
    ----------
    node_id : str
        Id of the node.
    in_col : str
        The name of the input column.
    out_col : str
        The name of the output column.
    """

    def __init__(self, node_id: str, in_col: str, out_col: str):
        super(Tokenizer, self).__init__(node_id)
        self.parameters = Parameters(
            inCol=KeyValueParameter("inputCol", str, in_col),
            outCol=KeyValueParameter("outputCol", str, out_col),
        )
        self.computational_instance = Tk(**self.parameters.get_dict())

    def execute(self):
        self.dataset = self.computational_instance.transform(self.dataset)


class HashingTF(Transformer):
    """Represent a Spark HashingTF that maps a sequence of terms to their term frequencies using the hashing trick.

    Input
    -----
    dataset : DataFrame
        A Spark DataFrame.

    Output
    ------
    dataset : DataFrame
        The modified Spark DataFrame.

    Parameters
    ----------
    node_id : str
        Id of the node.
    in_col : str
        The name of the input column.
    out_col : str
        The name of the output column.
    """

    def __init__(self, node_id: str, in_col: str, out_col: str):
        super(HashingTF, self).__init__(node_id)
        self.parameters = Parameters(
            inCol=KeyValueParameter("inputCol", str, in_col),
            outCol=KeyValueParameter("outputCol", str, out_col),
        )
        self.computational_instance = Htf(**self.parameters.get_dict())

    def execute(self):
        self.dataset = self.computational_instance.transform(self.dataset)


class LogisticRegression(Estimator):
    """Represent a SparkNode that supports fitting traditional logistic regression model.

    Input
    -----
    dataset : DataFrame
        A Spark DataFrame.

    Output
    ------
    model : PipelineModel
        A Spark PipelineModel.

    Parameters
    ----------
    max_iter : int
        Max number of iterations.

    reg_param : float
        Regularization parameter.
    """

    def __init__(self, node_id: str, max_iter: int, reg_param: float):
        super(LogisticRegression, self).__init__(node_id)
        self.parameters = Parameters(
            max_iter=KeyValueParameter("maxIter", int, max_iter),
            reg_param=KeyValueParameter("regParam", float, reg_param),
        )
        self.computational_instance = Lr(**self.parameters.get_dict())

    def execute(self):
        self.model = self.computational_instance.fit(self.dataset)
