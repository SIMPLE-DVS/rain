from simple_repo.parameter import KeyValueParameter, Parameters
from simple_repo.simple_spark.node_structure import Transformer, Estimator
from pyspark.ml.classification import LogisticRegression as Lr
from pyspark.ml.feature import HashingTF as Htf, Tokenizer as Tk


class Tokenizer(Transformer):
    """ Represent a Spark Tokenizer used to split text in individual term.

    Parameters
    ----------
    in_col: str
        The name of the input column

    out_col: str
        The name of the output column
    """

    def __init__(self, spark, in_col: str, out_col: str):
        self.parameters = Parameters(
            inCol=KeyValueParameter("inputCol", str, in_col),
            outCol=KeyValueParameter("outputCol", str, out_col)
        )
        super(Tokenizer, self).__init__(spark)

    def execute(self):
        return Tk(**self.parameters.get_dict())


class HashingTF(Transformer):
    """ Represent a Spark HashingTF that maps a sequence of terms to their term frequencies using the hashing trick.

    Parameters
    ----------
    in_col: str
        The name of the input column

    out_col: str
        The name of the output column
    """

    def __init__(self, spark, in_col: str, out_col: str):
        self.parameters = Parameters(
            inCol=KeyValueParameter("inputCol", str, in_col),
            outCol=KeyValueParameter("outputCol", str, out_col)
        )
        super(HashingTF, self).__init__(spark)

    def execute(self):
        return Htf(**self.parameters.get_dict())


class LogisticRegression(Estimator):
    """ Represent a SparkNode that supports fitting traditional logistic regression model.

    Parameters
    ----------
    max_iter: int

    reg_param: float
    """

    def __init__(self, spark, max_iter: int, reg_param: float):
        self.parameters = Parameters(
            max_iter=KeyValueParameter("maxIter", int, max_iter),
            reg_param=KeyValueParameter("regParam", float, reg_param)
        )
        super(LogisticRegression, self).__init__(spark)

    def execute(self):
        return Lr(**self.parameters.get_dict())
