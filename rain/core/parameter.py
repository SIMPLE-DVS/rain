from abc import abstractmethod
from typing import Any


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


class SimpleParameter:
    """Base class that represents a Parameter for a given node.

    Parameters
    ----------
    is_mandatory : bool, default False
        True if the parameter is mandatory, False otherwise.
    group_name : str, default None
        Name of the group of this parameter, used to pass it to the right function.
    """
    def __init__(self, is_mandatory: bool = False, group_name: str = None):
        self._is_mandatory = is_mandatory
        self.group_name = group_name

    @property
    def is_mandatory(self):
        """Returns the variable that specify if the parameter is mandatory."""
        return self._is_mandatory


class KeyValueParameter(SimpleParameter):
    """A KeyValue Parameter contains information about parameters that can be used during the transformation.

    Parameters
    ----------
    name : str
        The name of this parameter.
    p_type : type
        The type of this parameter.
    value : Any, default None
        The value of this parameter.
    is_mandatory : bool, default False
        True if the parameter is mandatory, False otherwise.
    """
    def __init__(
        self, name: str, p_type: type, value: Any = None, is_mandatory: bool = False
    ):
        self._name = name
        self._type = p_type
        self._value = value
        super(KeyValueParameter, self).__init__(is_mandatory)

    @property
    def name(self) -> str:
        """Returns the variable containing the name of the parameter."""
        return self._name

    @property
    def type(self) -> type:
        """Returns the variable containing the type of the parameter."""
        return self._type

    @property
    def value(self) -> Any:
        """Returns the variable containing the value of the parameter."""
        return self._value

    @value.setter
    def value(self, value):
        """Set the value of the parameter.

        Parameters
        ----------
        value : Any
            The value that should be set for this parameter
        """
        self._value = value

    def __str__(self):
        return "{{{}: {}}}".format(self._name, self._value)

    def __repr__(self):
        return "{{{}, {}, {}}}".format(
            self.value, self.type.__name__, self.is_mandatory
        )


class SimpleHyperParameter(SimpleParameter):
    """A KeyValue Parameter contains information about parameters that can be used during the transformation.

    Parameters
    ----------
    is_mandatory : bool, default False
        Name of the group of this parameter, used to pass it to the right function.
    """
    def __init__(self, is_mandatory: bool = False):
        super(SimpleHyperParameter, self).__init__(is_mandatory)

    @property
    def is_mandatory(self):
        """Returns the variable that specify if the parameter is mandatory."""
        return self._is_mandatory
