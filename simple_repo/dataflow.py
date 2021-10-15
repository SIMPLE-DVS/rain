from typing import Any, List
import networkx as nx
import copy

import simple_repo.base as base  # import module to avoid circular dependency
from simple_repo.exception import DuplicatedNodeId, EdgeConnectionError


class MultiEdge:
    """Represents an edge of the dataflow.

    Parameters
    ----------
    source : SimpleNode
        source node.
    destination : SimpleNode
        destination node.
    """

    def __init__(
        self,
        source: list,
        destination: list = None,
        source_output: List[str] = None,
        destination_input: List[str] = None,
    ):
        self.source = source
        self.destination = destination
        self.source_output = source_output
        self.destination_input = destination_input

    def __gt__(self, other):
        if isinstance(other, MultiEdge):
            self.destination = other.source
            self.destination_input = copy.deepcopy(other.source_output)
        elif not isinstance(other, base.OutputMixin):
            raise EdgeConnectionError("Node {} has no input variable.".format(other))
        else:
            self.destination = [other]
            self.destination_input = copy.deepcopy(self.source_output)

        return self

    def __and__(self, other):
        if isinstance(other, MultiEdge):
            self.source.extend(other.source)
            self.source_output = other.source_output
            return self
        elif isinstance(other, base.SimpleNode):
            self.source.append(other)
            return self

    def __matmul__(self, other):
        if type(other) is str:
            if not all(hasattr(node, other) for node in self.source):
                raise EdgeConnectionError(
                    "Node {} has no input called {}.".format(self.node_id, other)
                )
            self.source_output = [other]
        elif type(other) is list and all(type(item) is str for item in other):
            if not all(
                all(hasattr(node, string) for node in self.source) for string in other
            ):
                self.source_output = other
        else:
            raise EdgeConnectionError(
                "Unable to connect node {}. Node's variables must be specified as string or list of strings".format(
                    self.node_id
                )
            )

        return self


class DataFlow:
    def __init__(self, dataflow_id: str, executor: Any = None):
        self.id = dataflow_id
        self.executor = executor
        self._nodes = {}
        self._edges = []
        self._dag = nx.DiGraph(name=self.id)

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

    def add_edge(self, edge: MultiEdge):
        if edge.source is not None:
            for node in edge.source:
                try:
                    self.add_node(node)
                except DuplicatedNodeId:
                    pass

        if edge.destination is not None:
            for node in edge.destination:
                try:
                    self.add_node(node)
                except DuplicatedNodeId:
                    pass

        self._edges.append(edge)
        for dest_node in edge.destination:
            self._dag.add_edge(edge.source[0].node_id, dest_node.node_id)

    def add_edges(self, edges: List[MultiEdge]):
        for edge in edges:
            self.add_edge(edge)

    def get_edge(self, source, destination) -> MultiEdge:
        matching_edges = list(
            filter(
                lambda edge: source in edge.source and destination in edge.destination,
                self._edges,
            )
        )
        return matching_edges[0] if matching_edges else None

    def has_node(self, node):
        if type(node) is str:
            return lambda n: node in self._nodes.keys()

        return lambda n: node in self._nodes.values()

    def is_acyclic(self):
        return nx.is_directed_acyclic_graph(self._dag)
