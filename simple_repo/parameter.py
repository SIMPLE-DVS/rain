from abc import abstractmethod
from typing import Any

from simple_repo.exception import ParameterNotFound
from simple_repo.exception import BadParameterStructure


class SimpleIO:
    def __init__(self, io_type: type):
        self._type = io_type

    @property
    def type(self):
        return self._type


class SimpleParameter:
    def __init__(self, is_mandatory: bool = False):
        self._is_mandatory = is_mandatory

    @property
    def is_mandatory(self):
        return self._is_mandatory

    @abstractmethod
    def get_structure(self):
        pass


class KeyValueParameter(SimpleParameter):
    """
    A KeyValue Parameter contains information about parameters that can be used during the transformation.
    """

    def __init__(
        self, name: str, p_type: type, value: Any = None, is_mandatory: bool = False
    ):
        self._name = name
        self._type = p_type
        self._value = value
        super(KeyValueParameter, self).__init__(is_mandatory)

    def get_structure(self):
        struct = dict()

        struct["name"] = self._name
        struct["type"] = self._type.__name__
        struct["is_mandatory"] = self._is_mandatory
        return struct

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

    def __str__(self):
        return "{{{}: {}}}".format(self._name, self._value)

    def __repr__(self):
        return "{{{}, {}, {}}}".format(
            self.value, self.type.__name__, self.is_mandatory
        )


class StructuredParameterList(SimpleParameter):
    """
    Represent a parameter as a list of parameters all with the same structure.
    """

    def __init__(self, is_mandatory: bool = False, **keys_structure):
        # Save the structure of the parameters in two lists one for mandatory keys, one for optional keys
        self._mandatory_keys = []
        self._optional_keys = []

        for key, val in keys_structure.items():
            if val:
                self._mandatory_keys.append(key)
            elif not val:
                self._optional_keys.append(key)
            else:
                raise BadParameterStructure(
                    "Invalid assignment for parameter structure! Set True if the key is "
                    "mandatory, False otherwise."
                )

        # Save the list of parameters that will be eventually populated via setter
        self._parameters = []
        super(StructuredParameterList, self).__init__(is_mandatory)

    def get_structure(self):
        param = []
        param_struct = {}
        for k in self._mandatory_keys:
            param_struct["name"] = k
            param_struct["type"] = str.__name__
            param_struct["is_mandatory"] = True
            param.append(param_struct)
            param_struct = {}
        for k in self._optional_keys:
            param_struct["name"] = k
            param_struct["type"] = str.__name__
            param_struct["is_mandatory"] = False
            param.append(param_struct)
            param_struct = {}
        return param

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
            if param.get(key) is not None:
                new_param[key] = param.get(key)

        # the new parameter is added to the list of parameters
        self._parameters.append(new_param)

    def add_all_parameters(self, *params):
        for par in params:
            self.add_parameter(**par)

    @property
    def parameters(self):
        return self._parameters


class SimpleHyperParameter(SimpleParameter):
    def __init__(self, is_mandatory: bool = False):
        super(SimpleHyperParameter, self).__init__(is_mandatory)

    def get_structure(self):
        pass


if __name__ == "__main__":
    """
    [
        {
            col: ...
            pippo: ...
        },
        {
            col: ...
            alfio: ...
        }
    ]
    """
    p = StructuredParameterList(col=True, alfio=False, pippo=True)
    print(p.get_structure())

    k = KeyValueParameter("path", str)
    print(k.get_structure())
