from abc import abstractmethod
from typing import Any

from pyspark.sql import DataFrame
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession


class SparkParameter:
    def __init__(self, name: str, p_type: type, value: Any = None, is_required: bool = False):
        self._name = name
        self._value = value
        self._type = p_type
        self._is_required = is_required

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    @property
    def is_required(self):
        return self._is_required

    def __repr__(self):
        return "{{{}, {}, {}}}".format(self.value, self.type.__name__, self.is_required)


class SparkParameterList:

    def __init__(self, **keys_structure):
        self._mandatory_keys = []
        self._optional_keys = []

        for key, val in keys_structure.items():
            if val:
                self._mandatory_keys.append(key)
            elif not val:
                self._optional_keys.append(key)
            else:
                raise Exception(
                    "Invalid assignment for parameter structure! Set True if the key is mandatory,"
                    "False otherwise."
                )
        self._parameters = []

    def add_parameter(self, **param):
        new_param = {}
        for key in self._mandatory_keys:
            if key in param.keys():
                new_param[key] = param.get(key)
            else:
                raise Exception(
                    "Required parameter '{}' for class '{}' not found.".format(
                        key, self.__class__.__name__
                    )
                )
        for key in self._optional_keys:
            try:
                new_param[key] = param.get(key)
            except Exception:
                pass

        self._parameters.append(new_param)

    def add_all_parameters(self, *params):
        for par in params:
            self.add_parameter(**par)

    @property
    def parameters(self):
        return self._parameters


class SparkNode:
    _input = {
        "dataset": DataFrame,
        "spark": SparkSession
    }
    _attr = {}
    _output = {}

    def __init__(self, spark, **kwargs):
        for inp in self._input.keys():
            setattr(self, inp, None)

        self.spark = spark

        for out in self._output.keys():
            setattr(self, out, None)

        for k, v in kwargs.items():
            param = self._attr.get(k)
            if isinstance(param, SparkParameterList):
                param.add_all_parameters(*v)
            elif not isinstance(v, param.type):
                raise TypeError("Wrong type {} for {}".format(param.type, v))
            else:
                param.value = v

        print(self.__class__.__name__ + " Created with:")
        print("Node Input: {}".format(self._input))
        print("Node Output: {}".format(self._output))
        print("Node Attributes: {}\n".format(self._attr))

    def _get_attr_as_dict(self) -> dict:
        dct = {}
        for v in self._attr.values():
            dct[v.name] = v.value

        return dct

    @abstractmethod
    def execute(self):
        pass


class Transformer(SparkNode):

    _output = {
        "dataset": DataFrame
    }

    def __init__(self, spark, **kwargs):
        super(Transformer, self).__init__(spark, **kwargs)

    @abstractmethod
    def execute(self):
        pass


class Estimator(SparkNode):

    _output = {
        "model": PipelineModel
    }

    def __init__(self, spark, **kwargs):
        super(Estimator, self).__init__(spark, **kwargs)

    @abstractmethod
    def execute(self):
        pass
