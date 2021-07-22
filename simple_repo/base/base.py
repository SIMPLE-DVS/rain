from abc import abstractmethod

import pandas as pd
from typing import Any, List

_io_types_mapping = {"pandas.DataFrame": pd.DataFrame}

_types_mapping = {
    "str": str,
    "int": int,
    "float": float,
    "complex": complex,
    "list": list,
    "tuple": tuple,
    "range": range,
    "dict": dict,
    "set": set,
    "frozenset": frozenset,
    "bool": bool,
    "bytes": bytes,
    "bytearray": bytearray,
    "memoryview": memoryview,
}


class SimpleInput:
    def __init__(self, input_name: str, input_type: str):
        self._input_name = input_name
        self._input_type = _io_types_mapping[input_type]
        self._input_value = None

    @property
    def input_name(self):
        return self._input_name

    @property
    def input_type(self):
        return self._input_type

    @property
    def input_value(self):
        return self._input_value

    @input_value.setter
    def input_value(self, input_value: Any):
        self._input_value = input_value

    def __str__(self):
        return "{}: {}".format(self.__class__.__name__, self.__dict__)


class SimpleInputs:
    def __init__(self, inputs: List[dict]):
        self._inputs = [SimpleInput(**inp) for inp in inputs]

    def __str__(self):
        return "{}{}".format(
            self.__class__.__name__, [str(inp) for inp in self._inputs]
        )


class ParameterNotFound(Exception):
    def __init__(self, msg):
        self._msg = msg
        super(ParameterNotFound, self).__init__(self._msg)


class SimpleParameter:
    def __init__(
        self, param_name: str, param_type: str, is_mandatory: bool = False
    ) -> None:
        self._param_name = param_name
        self._param_type = _types_mapping[param_type]
        self._is_mandatory = is_mandatory
        self._param_value = None

    @property
    def param_name(self):
        return self._param_name

    @property
    def param_type(self):
        return self._param_type

    @property
    def param_value(self):
        return self._param_value

    @param_value.setter
    def param_value(self, param_value: Any):
        if not isinstance(param_value, (self._param_type,)):
            raise TypeError(
                "Expected type '{}' for parameter '{}', received '{}'.".format(
                    self._param_type.__name__,
                    self._param_name,
                    type(param_value).__name__,
                )
            )

        self._param_value = param_value

    def __str__(self):
        return "{}: {}".format(self.__class__.__name__, self.__dict__)


class SimpleParameters:
    def __init__(self, parameters: List[dict]):
        self._parameters = [SimpleParameter(**par) for par in parameters]

    def get_parameter(self, param_name: str):
        for sp in self._parameters:
            if sp.param_name == param_name:
                return sp

        return None

    def __str__(self):
        return "{}{}".format(
            self.__class__.__name__, [str(par) for par in self._parameters]
        )


class SimpleOutput:
    def __init__(self, output_name: str, output_type: str):
        self._output_name = output_name
        self._output_type = _io_types_mapping[output_type]
        self._output_value = None
        self._receiver_id = None
        self._receiver_field = None

    @property
    def output_name(self):
        return self._output_name

    @property
    def output_type(self):
        return self._output_type

    @property
    def output_value(self):
        return self._output_value

    @output_value.setter
    def output_value(self, output_value: Any):
        self._output_value = output_value

    @property
    def receiver_id(self):
        return self._receiver_id

    @receiver_id.setter
    def receiver_id(self, receiver_id: Any):
        self._receiver_id = receiver_id

    @property
    def receiver_field(self):
        return self._receiver_field

    @receiver_field.setter
    def receiver_field(self, receiver_field: Any):
        self._receiver_field = receiver_field

    def __str__(self):
        return "{}: {}".format(self.__class__.__name__, self.__dict__)


class SimpleOutputs:
    def __init__(self, outputs: List[dict]):
        self._outputs = [SimpleOutput(**out) for out in outputs]

    def get_output(self, output_name: str):
        for out in self._outputs:
            if out.output_name == output_name:
                return out

        return None

    def __str__(self):
        return "{}{}".format(
            self.__class__.__name__, [str(outs) for outs in self._outputs]
        )


class Meta(type):
    def __new__(cls, name, bases, dct):
        import json

        with open("structure.json", "r") as jf:
            jload = json.load(jf)

            super_new = super(Meta, cls).__new__

            parents = [b for b in bases if isinstance(b, Meta)]
            if not parents:
                dct["input"] = SimpleInputs(jload.get("input"))
                dct["parameters"] = SimpleParameters(jload.get("parameters"))
                dct["output"] = SimpleOutputs(jload.get("output"))

            return super_new(cls, name, bases, dct)


class SimpleStep(metaclass=Meta):
    def __init__(self, step_id: str, parameters: dict, output: list):
        super(SimpleStep, self).__init__()
        self._step_id = step_id
        self._init_parameters(**parameters)
        self._init_output(output)

    def _init_parameters(self, **kwargs):
        for k, v in kwargs.items():
            try:
                self.parameters.get_parameter(k).param_value = v
            except AttributeError:
                raise ParameterNotFound(
                    "Parameter {} can't be set for class {}.".format(
                        k, self.__class__.__name__
                    )
                )

    def _init_output(self, output_list: list):
        for output in output_list:
            try:
                out_obj = self.output.get_output(output.get("result"))
                out_obj.receiver_id = output.get("receiver_id")
                out_obj.receiver_field = output.get("receiver_field")
            except Exception as e:
                print(e)

    @abstractmethod
    def check_execution(self) -> bool:
        pass

    @abstractmethod
    def execute(self):
        pass

    @abstractmethod
    def communicate_result(self):
        pass

    @property
    def step_id(self):
        return self._step_id

    @step_id.setter
    def step_id(self, s_id: str):
        self._step_id = s_id

    def __str__(self):
        return "{}: {}{}{}".format(
            self.__class__.__name__,
            str(self.input),
            str(self.parameters),
            str(self.output),
        )


class SplitTrainTest(SimpleStep):
    def check_execution(self) -> bool:
        pass

    def execute(self):
        pass

    def communicate_result(self):
        pass


if __name__ == "__main__":
    params = {"train_size": 0.6}

    out = [
        {
            "result": "train_dataset",
            "receiver_id": "zscorepred",
            "receiver_field": "dataset",
        },
        {"result": "test_dataset", "receiver_id": "out", "receiver_field": "dataset"},
    ]

    stt = SplitTrainTest("stt", parameters=params, output=out)

    print(stt)
