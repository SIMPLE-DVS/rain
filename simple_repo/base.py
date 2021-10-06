import importlib
import json
from abc import abstractmethod
from typing import Any
import copy

import yaml


def get_class(fullname: str):
    """
    Given a fullname formed by "package + module + class" (a.e. sigmalib.load.loader.CSVLoader)
    imports dynamically the module and returns the wanted <class>
    """

    full_name_parts = fullname.split(".")

    package_name = ".".join(full_name_parts[:-2])
    module_name = full_name_parts[-2]
    class_name = full_name_parts[-1]

    if package_name != "":
        module = importlib.import_module("." + module_name, package_name)
    else:
        module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)

    return class_


def load_config(config_file) -> dict:
    """
    Utility function that given a path, returns the json file representing the configuration of the pipeline.
    """
    with open(config_file, "r") as f:
        config = json.load(f)
        return config


def load_yaml_config(config_file) -> dict:
    """
    Utility function that given a path, returns the yaml file representing the configuration of the pipeline.
    """
    with open(config_file, "r") as y:
        config = yaml.load(y, Loader=yaml.FullLoader)
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


class Meta(type):
    def __new__(mcs, clsname, bases, dct):
        input_vars_string = "_input_vars"
        output_vars_string = "_output_vars"
        methods_vars_string = "_methods"

        def get_new_input_vars():
            new_in_vars = {}
            new_in_vars = copy.deepcopy(bases[0]._input_vars)
            if len(bases) > 1:
                for index in range(1, len(bases)):
                    new_in_vars.update(bases[index]._input_vars)
            if dct.get(input_vars_string):
                new_in_vars.update(dct.get(input_vars_string))
            return new_in_vars

        def get_new_output_vars():
            new_out_vars = {}
            new_out_vars = copy.deepcopy(bases[0]._output_vars)
            if len(bases) > 1:
                for index in range(1, len(bases)):
                    new_out_vars.update(bases[index]._output_vars)
            if dct.get(output_vars_string):
                new_out_vars.update(dct.get(output_vars_string))
            return new_out_vars

        def get_new_methods():
            new_methods = {}
            new_methods = copy.deepcopy(bases[0]._methods)
            if len(bases) > 1:
                for index in range(1, len(bases)):
                    new_methods.update(bases[index]._methods)
            if dct.get(methods_vars_string):
                new_methods.update(dct.get(methods_vars_string))
            return new_methods

        if bases:
            if hasattr(bases[0], input_vars_string):
                in_vars = get_new_input_vars()
                dct.update({input_vars_string: in_vars})
            if hasattr(bases[0], output_vars_string):
                out_vars = get_new_output_vars()
                dct.update({output_vars_string: out_vars})
            if hasattr(bases[0], methods_vars_string):
                methods = get_new_methods()
                dct.update({methods_vars_string: methods})

        return super(Meta, mcs).__new__(mcs, clsname, bases, dct)


class SimpleNode(metaclass=Meta):

    def __init__(self):
        super(SimpleNode, self).__init__()

    @abstractmethod
    def execute(self):
        pass


class InputMixin:
    _output_vars = {}

    def __init__(self):
        # Set every output as an attribute if not already set
        for key in self._output_vars.keys():
            if not hasattr(self, key):
                setattr(self, key, None)

    def get_output_value(self, output_name: str):
        return vars(self).get(output_name)


class OutputMixin:
    _input_vars = {}

    def __init__(self):
        # Set every input as an attribute
        for key in self._input_vars.keys():
            if not hasattr(self, key):
                setattr(self, key, None)

    def set_input_value(self, input_name: str, input_value: Any):
        vars(self)[input_name] = input_value


class InputNode(SimpleNode, InputMixin):
    def __init__(self):
        super(InputNode, self).__init__()

    @abstractmethod
    def execute(self):
        pass


class ComputationalNode(SimpleNode, InputMixin, OutputMixin):

    def __init__(self):
        super(ComputationalNode, self).__init__()

    @abstractmethod
    def execute(self):
        pass


class OutputNode(SimpleNode, OutputMixin):
    def __init__(self):
        super(OutputNode, self).__init__()

    @abstractmethod
    def execute(self):
        pass


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


if __name__ == '__main__':
    pass
