from simple_repo.simple_spark.spark_node import Transformer, Estimator, SparkParameter
from pyspark.ml.classification import LogisticRegression as Lr
from pyspark.ml.feature import HashingTF as Htf, Tokenizer as Tk


class Tokenizer(Transformer):

    _attr = {
        "inCol": SparkParameter("inputCol", str, is_required=True),
        "outCol": SparkParameter("outputCol", str, is_required=True)
    }

    def __init__(self, spark, **kwargs):
        super(Tokenizer, self).__init__(spark, **kwargs)

    def execute(self):
        return Tk(**self._get_attr_as_dict())


class HashingTF(Transformer):

    _attr = {
        "inCol": SparkParameter("inputCol", str, is_required=True),
        "outCol": SparkParameter("outputCol", str, is_required=True)
    }

    def __init__(self, spark, **kwargs):
        super(HashingTF, self).__init__(spark, **kwargs)

    def execute(self):
        return Htf(**self._get_attr_as_dict())


class LogisticRegression(Estimator):

    _attr = {
        "maxIter": SparkParameter("maxIter", int, is_required=True),
        "regParam": SparkParameter("regParam", float, is_required=True)
    }

    def __init__(self, spark, **kwargs):
        super(LogisticRegression, self).__init__(spark, **kwargs)

    def execute(self):
        return Lr(**self._get_attr_as_dict())
