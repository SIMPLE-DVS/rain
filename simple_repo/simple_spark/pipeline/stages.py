from simple_repo.parameter import KeyValueParameter
from simple_repo.simple_spark.spark_node import Transformer, Estimator
from pyspark.ml.classification import LogisticRegression as Lr
from pyspark.ml.feature import HashingTF as Htf, Tokenizer as Tk


class Tokenizer(Transformer):
    _parameters = {
        "inCol": KeyValueParameter("inputCol", str, is_mandatory=True),
        "outCol": KeyValueParameter("outputCol", str, is_mandatory=True)
    }

    def __init__(self, spark, **kwargs):
        super(Tokenizer, self).__init__(spark, **kwargs)

    def execute(self):
        return Tk(**self._get_params_as_dict())


class HashingTF(Transformer):
    _parameters = {
        "inCol": KeyValueParameter("inputCol", str, is_mandatory=True),
        "outCol": KeyValueParameter("outputCol", str, is_mandatory=True)
    }

    def __init__(self, spark, **kwargs):
        super(HashingTF, self).__init__(spark, **kwargs)

    def execute(self):
        return Htf(**self._get_params_as_dict())


class LogisticRegression(Estimator):
    _parameters = {
        "maxIter": KeyValueParameter("maxIter", int, is_mandatory=True),
        "regParam": KeyValueParameter("regParam", float, is_mandatory=True)
    }

    def __init__(self, spark, **kwargs):
        super(LogisticRegression, self).__init__(spark, **kwargs)

    def execute(self):
        return Lr(**self._get_params_as_dict())
