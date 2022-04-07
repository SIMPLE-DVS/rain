import importlib
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Union
import copy

import networkx as nx

from rain.core.exception import (
    EdgeConnectionError,
    CyclicDataFlowException,
    DuplicatedNodeId,
)
from rain.core.execution import LocalExecutor
from rain.loguru_logger import logger


class LibTag(Enum):
    """
    Enumeration representing the library which the SimpleNode refers to
    """

    PANDAS = "Pandas"
    SPARK = "PySpark"
    SKLEARN = "Sklearn"
    MONGODB = "PyMongo"
    OTHER = "Other"
    BASE = "Base"


class TypeTag(Enum):
    """
    Enumeration representing the type of the SimpleNode according to its functionality
    """

    INPUT = "Input"
    OUTPUT = "Output"
    TRANSFORMER = "Transformer"
    CLASSIFIER = "Classifier"
    CLUSTERER = "CLusterer"
    REGRESSOR = "Regressor"
    ESTIMATOR = "Estimator"
    METRICS = "Metrics"
    OTHER = "Other"
    CUSTOM = "Custom"


@dataclass
class Tags:
    """
    DataClass that acts as a tag for a SimpleNode: it stores the library and the type of the node
    """

    library: LibTag
    type: TypeTag


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

        has_input = bool(bases_w_in_vars)  # check if bases_w_in_vars is empty

        new_in_vars_dct = {}
        for base in bases_w_in_vars:
            new_in_vars_dct.update(copy.deepcopy(base._input_vars))

        if input_vars_string in dct.keys():
            new_in_vars_dct.update(copy.deepcopy(dct.get(input_vars_string)))
            has_input = True

        if has_input:
            dct[input_vars_string] = new_in_vars_dct
            for (
                param
            ) in (
                new_in_vars_dct.keys()
            ):  # set attributes named as the keys in the _input_vars dictionary
                dct[param] = None

        # dct[output_vars_string] = union between dct["_output_vars"] if exist
        # and all the _output_vars those parents that have it.

        bases_w_out_vars = list(
            filter(lambda base: hasattr(base, output_vars_string), bases)
        )

        has_output = bool(bases_w_out_vars)  # check if bases_w_out_vars is empty

        new_out_vars_dct = {}
        for base in bases_w_out_vars:
            new_out_vars_dct.update(copy.deepcopy(base._output_vars))

        if output_vars_string in dct.keys():
            new_out_vars_dct.update(copy.deepcopy(dct.get(output_vars_string)))
            has_output = True

        if has_output:
            dct[output_vars_string] = new_out_vars_dct
            for (
                param
            ) in (
                new_out_vars_dct.keys()
            ):  # set attributes named as the keys in the _output_vars dictionary
                dct[param] = None

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
        logger.info("Create Node", node_name=type(self).__name__)

        super(SimpleNode, self).__init__()
        self.node_id = node_id

    @abstractmethod
    def execute(self):
        pass

    @abstractmethod
    def has_attribute(self, attribute: str) -> bool:
        pass

    def __hash__(self):
        return hash(self.node_id)

    def __eq__(self, other):
        if not isinstance(other, SimpleNode):
            return False

        if not self.node_id == other.node_id:
            return False

        return True

    def __matmul__(self, other):
        return EdgeContentSpecifier(self, other)

    def __str__(self):
        return self.node_id


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

    def has_attribute(self, attribute: str) -> bool:
        return attribute in self._output_vars.keys()

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.OTHER, TypeTag.INPUT)


class ComputationalNode(SimpleNode, InputMixin, OutputMixin):
    def __init__(self, node_id: str):
        super(ComputationalNode, self).__init__(node_id)

    @abstractmethod
    def execute(self):
        pass

    def has_attribute(self, attribute: str) -> bool:
        in_out_vars = set(self._input_vars.keys()).union(self._output_vars.keys())
        return attribute in in_out_vars


class OutputNode(SimpleNode, OutputMixin):
    def __init__(self, node_id: str):
        super(OutputNode, self).__init__(node_id)

    @abstractmethod
    def execute(self):
        pass

    def has_attribute(self, attribute: str) -> bool:
        return attribute in self._input_vars.keys()

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.OTHER, TypeTag.OUTPUT)


class EdgeContentSpecifier:
    """It works as an attribute specifier for the nodes that are used within a Multiedge.

    Parameters
    ----------
    node : SimpleNode
        The node that contains the chosen attributes
    nodes_attributes : Union[str, List]
        The chosen attributes of the node, they can either be the input or the output of the node.
    """

    def __init__(self, node: SimpleNode, nodes_attributes: Union[str, List]):
        self.node = node

        if isinstance(nodes_attributes, str):
            nodes_attributes = [nodes_attributes]
        elif (
            not isinstance(nodes_attributes, list)
            or not bool(list)
            or not all([isinstance(attr, str) for attr in nodes_attributes])
        ):
            raise EdgeConnectionError(
                f"The chosen {nodes_attributes} node attributes must be either string or a non-empty list."
            )

        for attr in nodes_attributes:
            if not node.has_attribute(attr):
                raise EdgeConnectionError(f"Node {node} has no attribute {attr}")

        self.nodes_attributes = nodes_attributes

    def __gt__(self, other):
        if not isinstance(other, EdgeContentSpecifier):
            raise EdgeConnectionError(
                "The right side of '>' must be an EdgeContentSpecifier!"
                "The latter can be created with 'node_var @ ['in_out_attr']'"
            )

        return MultiEdge(self, other)


class MultiEdge:
    """Represents an edge of the dataflow.

    Parameters
    ----------
    source : EdgeContentSpecifier
        source node and its attributes.
    destination : EdgeContentSpecifier
        destination node and its attributes.
    """

    def __init__(
        self,
        source: EdgeContentSpecifier,
        destination: EdgeContentSpecifier,
    ):
        if not isinstance(source.node, InputMixin):
            raise EdgeConnectionError(f"Node '{source.node}' has no output variable.")
        elif not isinstance(destination.node, OutputMixin):
            raise EdgeConnectionError(
                f"Node '{destination.node}' has no input variable."
            )

        logger.debug(
            f"Create edge from {source.nodes_attributes} to {destination.node.node_id} - {destination.nodes_attributes}",
            node_name=type(source.node).__name__,
        )

        self.source = source
        self.destination = destination


class DataFlow:
    def __init__(self, dataflow_id: str, executor: Any = LocalExecutor()):
        logger.info("Create Dataflow", dataflow_id=dataflow_id)

        self.id = dataflow_id
        self.executor = executor
        self._nodes = {}
        self._edges: List[MultiEdge] = []
        self._dag = nx.DiGraph(name=self.id)

        logger.debug(
            f"Use executor {type(self.executor).__name__}", dataflow_id=dataflow_id
        )

    def add_node(self, node) -> bool:
        """Add a node to the dataflow. If a node with the same node id exists then an exception will be raised.

        Parameters
        ----------
        node : SimpleNode
            The node to add.

        Returns
        -------
        bool
            True if the node has been correctly added.

        Raises
        ------
        DuplicatedNodeId
            If a node with the same node id already exists.

        """
        logger.info(f"Add node {type(node).__name__}", dataflow_id=self.id)
        if node.node_id in self._nodes.keys():
            raise DuplicatedNodeId(
                "The node identified as {} already exists within the DataFlow.".format(
                    node.node_id
                )
            )

        self._nodes[node.node_id] = node
        self._dag.add_node(node.node_id)

        return True

    def add_nodes(self, nodes) -> bool:
        """Add a node to the dataflow. If a node with the same node id exists then an exception will be raised.

        Parameters
        ----------
        nodes : list of SimpleNode
            The node to add.

        Returns
        -------
        bool
            True if the node has been correctly added.

        Raises
        ------
        DuplicatedNodeId
            If a node with the same node id already exists.

        """
        for node in nodes:
            self.add_node(node)

        return True

    def get_node(self, node_id: str):
        return self._nodes.get(node_id) if node_id in self._nodes.keys() else None

    def add_edge(self, edge: MultiEdge):
        logger.info(
            f"Add edge from {edge.source.node.node_id} - {edge.source.nodes_attributes} to {edge.destination.node.node_id} - {edge.destination.nodes_attributes}",
            dataflow_id=self.id,
        )
        try:
            self.add_node(edge.source.node)
        except DuplicatedNodeId:
            pass

        try:
            self.add_node(edge.destination.node)
        except DuplicatedNodeId:
            pass

        self._edges.append(edge)
        self._dag.add_edge(edge.source.node.node_id, edge.destination.node.node_id)

    def add_edges(self, edges: List[MultiEdge]):
        for edge in edges:
            self.add_edge(edge)

    def get_edge(self, source: SimpleNode, destination: SimpleNode) -> MultiEdge:
        matching_edges = list(
            filter(
                lambda edge: source == edge.source.node
                and destination == edge.destination.node,
                self._edges,
            )
        )
        return matching_edges[0] if matching_edges else None

    def get_outgoing_edges(self, source: SimpleNode) -> List[MultiEdge]:
        matching_edges = list(
            filter(
                lambda edge: source == edge.source.node,
                self._edges,
            )
        )
        return matching_edges

    def get_ingoing_edges(self, destination):
        matching_edges = list(
            filter(
                lambda edge: destination in edge.destination,
                self._edges,
            )
        )
        return matching_edges

    def has_node(self, node):
        if type(node) is str:
            return lambda n: node in self._nodes.keys()

        return lambda n: node in self._nodes.values()

    def is_acyclic(self):
        return nx.is_directed_acyclic_graph(self._dag)

    def get_execution_ordered_nodes(self):
        topological_order = nx.topological_sort(self._dag)
        return list(map(lambda node_id: self._nodes.get(node_id), topological_order))

    def execute(self):
        if not self.is_acyclic():
            raise CyclicDataFlowException(self.id)
        logger.info("Start execution of the Dataflow", dataflow_id=self.id)
        self.executor.execute(self)
