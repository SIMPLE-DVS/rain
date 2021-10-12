import importlib
from abc import abstractmethod
from typing import Any
import copy


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


def reset(simple_node):
    dic = vars(simple_node)
    for i in dic.keys():
        dic[i] = None


class Meta(type):
    def __new__(mcs, clsname, bases, dct):
        input_vars_string = "_input_vars"
        output_vars_string = "_output_vars"
        methods_vars_string = "_methods"

        # dct[input_vars_string] = union between dct["_input_vars"] if exist
        # and all the _input_vars those parents that have it.

        bases_w_in_vars = list(
            filter(lambda base: hasattr(base, input_vars_string), bases)
        )

        has_input = bool(bases_w_in_vars)  # check if bases_w_meth_vars is empty

        new_in_vars_dct = {}
        for base in bases_w_in_vars:
            new_in_vars_dct.update(copy.deepcopy(base._input_vars))

        if input_vars_string in dct.keys():
            new_in_vars_dct.update(copy.deepcopy(dct.get(input_vars_string)))
            has_input = True

        if has_input:
            dct[input_vars_string] = new_in_vars_dct

        # dct[output_vars_string] = union between dct["_output_vars"] if exist
        # and all the _output_vars those parents that have it.

        bases_w_out_vars = list(
            filter(lambda base: hasattr(base, output_vars_string), bases)
        )

        has_output = bool(bases_w_out_vars)  # check if bases_w_meth_vars is empty

        new_out_vars_dct = {}
        for base in bases_w_out_vars:
            new_out_vars_dct.update(copy.deepcopy(base._output_vars))

        if output_vars_string in dct.keys():
            new_out_vars_dct.update(copy.deepcopy(dct.get(output_vars_string)))
            has_output = True

        if has_output:
            dct[output_vars_string] = new_out_vars_dct

        # dct[methods_vars_string] = union between dct["_methods"] if exist
        # and all the _methods those parents that have it.

        bases_w_meth_vars = list(
            filter(lambda base: hasattr(base, methods_vars_string), bases)
        )

        has_methods = bool(bases_w_meth_vars)  # check if bases_w_meth_vars is empty

        new_meth_vars_dct = {}
        for base in bases_w_meth_vars:
            new_meth_vars_dct.update(copy.deepcopy(base._methods))

        if methods_vars_string in dct.keys():
            new_meth_vars_dct.update(copy.deepcopy(dct.get(methods_vars_string)))
            has_methods = True

        if has_methods:
            dct[methods_vars_string] = new_meth_vars_dct

        return super().__new__(mcs, clsname, bases, dct)


class SimpleNode(metaclass=Meta):
    def __init__(self, node_id: str):
        super(SimpleNode, self).__init__()
        self.node_id = node_id

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
    def __init__(self, node_id: str):
        super(InputNode, self).__init__(node_id)

    @abstractmethod
    def execute(self):
        pass


class ComputationalNode(SimpleNode, InputMixin, OutputMixin):
    def __init__(self, node_id: str):
        super(ComputationalNode, self).__init__(node_id)

    @abstractmethod
    def execute(self):
        pass


class OutputNode(SimpleNode, OutputMixin):
    def __init__(self, node_id: str):
        super(OutputNode, self).__init__(node_id)

    @abstractmethod
    def execute(self):
        pass


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
