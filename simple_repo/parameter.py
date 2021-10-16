from abc import abstractmethod
from typing import Any

from simple_repo.exception import ParameterNotFound
from simple_repo.exception import BadParameterStructure


class Parameters:
    """Parameters handles all the parameters within a SimpleNode.

    It gives the possibility to add one or several parameters,
    group parameters together, retrieve parameters and get
    a dictionary representation of the parameters useful to
    pass them to library functions as kwargs.
    """

    def __init__(self, **kwargs):
        # Set every parameter as an attribute
        self.pars = kwargs
        for key, val in kwargs.items():
            setattr(self, key, val)

    def add_parameter(self, parameter_name: str, parameter):
        """Add a parameter in the collection.

        Parameters
        ----------
        parameter_name : str
            Name of the parameter, can be used later to reference it as an attribute.
        parameter : SimpleParameter
            The parameter to add.
        """
        self.pars[parameter_name] = parameter
        setattr(self, parameter_name, parameter)

    def add_all_parameters(self, **kwargs):
        """Add one or more parameter in the collection.

        Parameters
        ----------
        kwargs : dict
            Of the form {param_name: parameter}. Each key will be set as the attribute name.
        """
        for key, val in kwargs.items():
            self.add_parameter(key, val)

    def add_group(self, group_name: str, keys: list):
        """Adds a group name to some parameters.

        Parameters
        ----------
        group_name : str
            Name of the group.
        keys : list[str]
            Used to specify the parameters to include in the group.
            Each string must correspond to the attribute name of the parameter.
        """
        for param_name in keys:
            if param_name in self.pars and isinstance(
                self.pars[param_name], KeyValueParameter
            ):
                self.pars[param_name].group_name = group_name

    def group_all(self, group_name: str):
        """Adds a group name to all the parameters.

        Parameters
        ----------
        group_name : str
            Name of the group.
        """
        self.add_group(group_name, list(self.pars.keys()))

    def get_all(self):
        """Gets all the parameters.

        Returns
        -------
        list[SimpleParameter]
        """
        return list(self.pars.values())

    def get_all_from_group(self, group_name: str):
        """Gets all the parameters contained in a group.

        Parameters
        ----------
        group_name : str
            Name of the group.

        Returns
        -------
        list[SimpleParameter]
        """
        return list(
            filter(
                lambda elem: elem.group_name is not None
                and elem.group_name == group_name,
                self.pars.values(),
            )
        )

    def get_dict(self):
        """Gets all the KeyValueParameters as a dictionary, in order to simplify passing parameters to library functions.

        Returns
        ----------
        dict[str, Any]
            dict of the form {param_lib_name, param_value} where the key is the name of the parameter as required from
            the library.
        """
        parameters = dict(
            (par.name, par.value)
            for par in self.pars.values()
            if isinstance(par, KeyValueParameter)
        )
        return parameters

    def get_dict_from_group(self, group_name: str):
        """Gets all the KeyValueParameters contained in a group as a dictionary, in order to simplify passing parameters to library functions.

        Returns
        ----------
        dict[str, Any]
            dict of the form {param_lib_name, param_value} where the key is the name of the parameter as required from
            the library.
        """
        parameters = dict(
            (par.name, par.value) for par in self.get_all_from_group(group_name)
        )
        return parameters


class SimpleIO:
    def __init__(self, io_type: type):
        self._type = io_type

    @property
    def type(self):
        return self._type


class SimpleParameter:
    def __init__(self, is_mandatory: bool = False, group_name: str = None):
        self._is_mandatory = is_mandatory
        self.group_name = group_name

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
            if type(val) is bool:
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
        invalid_keys = list(
            filter(
                lambda k: k not in self._mandatory_keys
                and k not in self._optional_keys,
                param.keys(),
            )
        )

        if invalid_keys:
            raise ParameterNotFound(
                "Invalid parameters keys {}. The key is neither mandatory nor optional.".format(
                    invalid_keys
                )
            )

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

    def has_parameters(self, **kwargs):
        for key, val in kwargs.items():
            if not any(
                map(
                    lambda par: True
                    if key in par.keys() and val == par.get(key)
                    else False,
                    self._parameters,
                )
            ):
                return False

        return True


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
