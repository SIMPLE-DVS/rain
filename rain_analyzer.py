import json

from setuptools import find_packages
import importlib
import inspect
import sys

from pkgutil import iter_modules
import re

from rain import SimpleNode

from numpydoc.docscrape import ClassDoc


class SimpleNodeInfo:
    """
    Class representing a SimpleNode: it contains all the node information that the GUI needs to display.

    Parameters
    ----------

    node_class: str
        The name of the SimpleNode class
    node_package: str
        The package of the SimpleNode class
    node_input: dict
        The inputs that the SimpleNode take
    node_output: dict
        The outputs that the SimpleNode returns
    node_parameter: list
        The list of the SimpleNode parameters
    node_methods: list
        The list of the SimpleNode methods that can be executed
    node_tags: dict
        The tag containing the library and type of the SimpleNode
    node_name: str
        The name of the SimpleNode (it is a different representation of the node_class)
    node_description: str
        The description of the SimpleNode class
    """

    def __init__(
        self,
        node_class: str,
        node_package: str,
        node_input: dict,
        node_output: dict,
        node_parameter: list,
        node_methods: list,
        node_tags: dict,
        node_name: str,
        node_description: str,
    ):
        self.clazz = node_class
        self.package = node_package
        self.input = node_input
        self.output = node_output
        self.parameter = node_parameter
        self.methods = node_methods
        self.tags = node_tags
        self.name = node_name
        self.description = node_description


def find_modules(lib_path):
    """
    Given a path of a directory, return all the python module contained within the directory itself.
    """
    modules_list = set()
    for package in find_packages(lib_path):
        pkg_path = lib_path + "/" + package.replace(".", "/")
        if sys.version_info.major == 2 or (
            sys.version_info.major == 3 and sys.version_info.minor < 6
        ):
            for _, m_name, is_pkg in iter_modules([pkg_path]):
                if not is_pkg:
                    modules_list.add(package + "." + m_name)
        else:
            for info in iter_modules([pkg_path]):
                if not info.ispkg:
                    modules_list.add(package + "." + info.name)
    return modules_list


def get_simple_node_subclasses(simple_modules: set):
    """
    Given a list of python modules it return a set containing all the classes that are SimpleNode with no child.
    """
    parent_classes = set()
    child_classes = set()

    for m in simple_modules:
        module = importlib.import_module("rain.{}".format(m))
        for name, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, SimpleNode):
                parent = cls.__base__
                if parent in child_classes:
                    child_classes.remove(parent)
                parent_classes.add(parent)
                if cls not in parent_classes:
                    child_classes.add(cls)
    return child_classes


def get_io_structure(io_dict: dict):
    """
    Given a dict containing the information about the input or output of a SimpleNode, return a new dict containing
    the same information in a more readable format.
    """
    io = {}
    for k, v in io_dict.items():
        io[k] = v.__name__
    return io


def get_class(cls, package="simple_repo.", module=".node_structure"):
    """
    Given a module and a package string it returns the required class in that module.
    """
    module = importlib.import_module(module, package=package)
    clazz = getattr(module, cls)
    return clazz


def get_simple_nodes_info(node_subclasses: set):
    """
    Given a list of SimpleNode classes it returns the list containing all the information about them.
    """
    simple_nodes = []
    for cls in node_subclasses:
        docstring = ClassDoc(cls)
        cls_description = " ".join(docstring._parsed_data.get("Summary"))
        params_desc, params_type = extract_params_info(
            docstring._parsed_data.get("Parameters")
        )
        parameters = get_parameters(cls, params_desc, params_type)
        methods = list(cls._methods.keys()) if hasattr(cls, "_methods") else None
        in_vars = (
            get_io_structure(cls._input_vars) if hasattr(cls, "_input_vars") else None
        )
        out_vars = (
            get_io_structure(cls._output_vars) if hasattr(cls, "_output_vars") else None
        )
        name = " ".join(
            re.sub(
                "([A-Z][a-z]+)", r" \1", re.sub("([A-Z]+)", r" \1", cls.__name__)
            ).split()
        )
        node_type = {
            "library": cls._get_tags().library.value,
            "type": cls._get_tags().type.value,
        }

        node_info = SimpleNodeInfo(
            cls.__name__,
            cls.__module__ + "." + cls.__name__,
            in_vars,
            out_vars,
            parameters,
            methods,
            node_type,
            name,
            cls_description,
        )
        simple_nodes.append(node_info)
    return simple_nodes


def extract_params_info(parameters):
    """
    Given a list of parameters in numpyDoc format, it return two dict containing the information about the
    description and the type of each parameter.
    """
    params_desc = {}
    params_type = {}
    for p in parameters:
        params_desc[p.name] = " ".join(p.desc)
        if ", default" in p.type:
            params_type[p.name] = p.type.split(", default")[0].strip()
        else:
            params_type[p.name] = p.type

    return params_desc, params_type


def get_parameters(cls, params_desc, params_type):
    """
    Given a SimpleNode class and the information about its parameters, it returns a list containing the name,
    the type, the default_value, the description and if it is mandatory for each parameter.
    """
    cls_parameters = list(inspect.signature(cls.__init__).parameters.values())[2:]
    parameters = []
    for p in cls_parameters:
        name = p.name
        p_type = params_type.get(name) if params_desc.get(name) else None
        is_mandatory = True if p.default is inspect.Signature.empty else False
        default_value = p.default if not is_mandatory else None
        description = params_desc.get(name) if params_desc.get(name) else None
        parameters.append(
            {
                "name": name,
                "type": p_type,
                "is_mandatory": is_mandatory,
                "default_value": default_value,
                "description": description,
            }
        )
    return parameters


def get_libraries(classes):
    libs = set()
    for clz in classes:
        libs.add(clz._get_tags().library.value)
    return libs


def get_dependencies(libs):
    pass


def analyze():
    """
    Analyze the Rain library searching for all the classes that are SimpleNode. Return a list of object
    containing all the relevant information about the classes.
    """
    modules = find_modules("./rain")

    simple_node_subclasses = get_simple_node_subclasses(modules)

    libs = get_libraries(simple_node_subclasses)
    # libs = get_dependencies(libs)

    simple_nodes_info = get_simple_nodes_info(simple_node_subclasses)
    simple_nodes_info = [node.__dict__ for node in simple_nodes_info]

    info = {"nodes": simple_nodes_info, "dependencies": libs}

    with open("./analyzer_output/rain_structure.json", "w") as f:
        json.dump(info, f)

    return simple_nodes_info


def get_analyzed_nodes():
    with open("./analyzer_output/rain_structure.json", "r") as f:
        nodes = json.load(f)
    return nodes


if __name__ == "__main__":
    n = analyze()
    print("ok")
