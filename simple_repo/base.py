import importlib
import json
from abc import abstractmethod
from typing import Any

from simple_repo.exception import ParameterNotFound
from simple_repo.parameter import StructuredParameterList


def get_class(fullname: str):
    """
    Given a fullname formed by "package + module + class" (a.e. sigmalib.load.loader.CSVLoader)
    imports dynamically the module and returns the wanted <class>
    """

    full_name_parts = fullname.split(".")

    package_name = ".".join(full_name_parts[:-2])
    module_name = full_name_parts[-2]
    class_name = full_name_parts[-1]

    module = importlib.import_module("." + module_name, package_name)
    class_ = getattr(module, class_name)

    return class_


def load_config(config_file) -> dict:
    """
    Utility function that given a path, returns the json file representing the configuration of the pipeline.
    """
    with open(config_file, "r") as f:
        config = json.load(f)
        return config


def get_step(step_id: str, step_list: list):
    """
    Utility function that given a step id and the list of steps, returns the step instance with the corresponding id.
    """
    # get all the nodes with the request id
    corr_steps = [step for step in step_list if step.step_id == step_id]

    if len(corr_steps) > 1:
        raise Exception(
            "Error! There are duplicated nodes with same id '{}'.".format(step_id)
        )
    elif len(corr_steps) < 1:
        raise Exception("Error! There aren't nodes with id '{}'.".format(step_id))

    return corr_steps[0]


def reset(simple_node):
    dic = vars(simple_node)
    for i in dic.keys():
        dic[i] = None


class SimpleNode:
    _input_vars = {}
    _parameters = {}
    _output_vars = {}

    def __init__(self, **kwargs):

        # Set every input as an attribute
        for key in self._input_vars.keys():
            setattr(self, key, None)

        # Set every output as an attribute if not already set
        for key in self._output_vars.keys():
            if key not in self._input_vars:
                setattr(self, key, None)

        # check the parameter passed and set their values
        for name, value in kwargs.items():
            try:
                # retrieve the parameter from its name
                par = self._parameters.get(name)

                # if it is a parameter list add all the values inside, otherwise set the value of the parameter.
                if isinstance(par, StructuredParameterList):
                    par.add_all_parameters(*value)
                elif not isinstance(value, par.type):
                    raise TypeError(
                        "Expected type '{}' for parameter '{}' in class '{}', received type '{}'.".format(
                            par.type, name, self.__class__.__name__, type(value)
                        )
                    )
                else:
                    par.value = value

            except AttributeError:
                raise ParameterNotFound(
                    "Class '{}' has no attribute '{}'".format(
                        self.__class__.__name__, name
                    )
                )

    def _get_params_as_dict(self) -> dict:
        dct = {}
        for val in self._parameters.values():
            dct[val.name] = val.value

        return dct

    def set_input_value(self, input_name: str, input_value: Any):
        vars(self)[input_name] = input_value

    def get_output_value(self, output_name: str):
        return vars(self).get(output_name)

    @abstractmethod
    def execute(self):
        pass


class Node(object):
    """
    Class to represent, as a Python object, the configuration file.

        Parameters
        ----------

        idd : string
            The unique identifier that each node must have.

        name : string
            The full-name formed by \textit{package + module + class}, useful to dynamically import the
            module and to return the wanted class representing one step of the pipeline

        attr : dict
            List of features that characterizes each step of the pipeline. Obviously, depending on the node,
            we have a different structure of the list with different number of features.

        then : list
            List of idd representing the node(s) that are directly linked with the current node.

    """

    def __init__(
        self,
        node_id: str,
        node: str,
        node_type: str,
        parameters: dict,
        then: list = None,
        execute: list = None,
    ):
        self._node_id = node_id
        self._node = node
        self._node_type = node_type
        self._parameters = parameters
        self._execute = execute
        self._then = then

    @property
    def node_id(self):
        return self._node_id

    @property
    def node(self):
        return self._node

    @property
    def node_type(self):
        return self._node_type

    @property
    def parameters(self):
        return self._parameters

    @property
    def execute(self):
        return self._execute

    @property
    def then(self):
        return self._then


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
