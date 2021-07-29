import csv
import inspect
from abc import ABC, abstractmethod
import functools
import pandas as pd
from typing import Any, List


class ParameterNotFound(Exception):
    def __init__(self, msg: str):
        super(ParameterNotFound, self).__init__(msg)


class PandasParameter:
    """
    A Pandas Parameter contains information about parameters that can be used during the transformation.
    """

    def __init__(
        self,
        param_name: str,
        param_type: type,
        param_value: Any = None,
        is_mandatory: bool = False,
    ):
        self._param_name = param_name
        self._param_type = param_type
        self._param_value = param_value
        self._is_mandatory = is_mandatory

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
    def param_value(self, param_value):
        self._param_value = param_value

    def __str__(self):
        return "{{{}: {}}}".format(self._param_name, self._param_value)


class PandasParameterList:
    """
    Represent a parameter as a list of parameters all with the same structure.
    """

    def __init__(self, **keys_structure):
        # Save the structure of the parameters in two lists
        # one for mandatory keys, one for optional keys
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

        # Save the list of parameters the will be eventually populated via setter
        self._parameters = []

    def add_parameter(self, **param):
        new_param = {}

        # check if all the mandatory keys are present and add their value
        for key in self._mandatory_keys:
            if key in param.keys():
                new_param[key] = param.get(key)
            else:
                raise ParameterNotFound(
                    "Required parameter '{}' for class '{}' not found.".format(
                        key, self.__class__.__name__
                    )
                )

        # if any of the optional keys is present its value is added
        for key in self._optional_keys:
            try:
                new_param[key] = param.get(key)
            except Exception:
                pass

        # the new parameter is added to the list of parameters
        self._parameters.append(new_param)

    def add_all_parameters(self, *params):
        for par in params:
            self.add_parameter(**par)

    @property
    def parameters(self):
        return self._parameters


class PandasNode:
    """
    Every PandasNode takes a pandas DataFrame as input,
    applies a tansformation and returns a pandas DataFrame as output.
    """

    _input_vars = {"dataset": pd.DataFrame}
    _parameters = {}
    _output_vars = {"dataset": pd.DataFrame}

    def _get_params_as_dict(self) -> dict:
        dct = {}
        for pval in self._parameters.values():
            dct[pval.param_name] = pval.param_value

        return dct

    def __init__(self, **kwargs):

        # Set every input as an attribute
        for key in self._input_vars.keys():
            setattr(self, key, None)

        # Set every output as an attribute if not already set
        for key in self._output_vars.keys():
            if key not in self._input_vars:
                setattr(self, key, None)

        # check the parameter passed and set their values
        for param_inst_name, param_inst_val in kwargs.items():
            try:
                # retrieve the parameter from its name
                par = self._parameters.get(param_inst_name)

                # if it is a parameter list add all the values inside, otherwise set the value of the parameter.
                if isinstance(par, PandasParameterList):
                    par.add_all_parameters(*param_inst_val)
                elif not isinstance(param_inst_val, par.param_type):
                    raise TypeError(
                        "Expected type '{}' for parameter '{}' in class '{}', received type '{}'.".format(
                            par.param_type,
                            param_inst_name,
                            self.__class__.__name__,
                            type(param_inst_val),
                        )
                    )
                else:
                    par.param_value = param_inst_val

            except AttributeError:
                raise ParameterNotFound(
                    "Class '{}' has no attribute '{}'".format(
                        self.__class__.__name__, param_inst_name
                    )
                )

    @abstractmethod
    def execute(self):
        pass

    def __str__(self):
        return "{}".format(self._get_params_as_dict())


class PandasPipeline:
    """
    PandasPipeline represents a sequence of transformation of a pandas dataframe.
    The nodes to use for the transformation are sent in a list of stages.
    The method transform is used to start the computation.
    """

    def __init__(self, stages: List[PandasNode]):
        self._stages = stages

    @property
    def stages(self):
        return self._stages

    def append_stage(self, stage: PandasNode):
        self._stages.append(stage)

    def execute(self):
        if len(self._stages) == 0:
            return None

        for i in range(0, len(self._stages)):
            self._stages[i].execute()

            if i + 1 > len(self._stages) - 1:
                break

            try:
                self._stages[i + 1].dataset = self._stages[i].dataset
                self._stages[i].dataset = None
            except Exception as e:
                print(e)

        return self._stages[len(self._stages) - 1].dataset
