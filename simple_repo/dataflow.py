from typing import Any, List
import networkx as nx

from simple_repo.exception import DuplicatedNodeId


class DataFlow:
    def __init__(self, dataflow_id: str, executor: Any = None):
        self.id = dataflow_id
        self.executor = executor
        self._nodes = {}
        self._dag = nx.MultiDiGraph(name=self.id)

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
        self._dag.add_node(node.node_id, node=node)

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

    def is_acyclic(self):
        return nx.is_directed_acyclic_graph(self._dag)


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
        source,
        destination=None,
        source_output: List[str] = None,
        destination_input: List[str] = None,
    ):
        self.source = source
        self.destination = destination
        self.source_output = source_output
        self.destination_input = destination_input
