from typing import List

from simple_repo.base import Node
import networkx as nx
from matplotlib import pyplot as plt


def get_edges(node_list):
    edges = []
    for node in node_list.values():
        if node.then is not None:
            for nxt in node.then:
                edges.append((node.node_id, nxt.get("send_to")))
    return edges


class DagCreator:
    """
    Takes a list of Node and parse it to a DAG.
    """

    def __init__(self):
        self._graph = nx.DiGraph()
        self._is_loaded = False
        self._nodes = {}
        self._sorted_nodes = None

    def create_dag(self, node_list: list):
        edges = get_edges(node_list)
        self._nodes.update(node_list)
        self._graph.add_edges_from(edges)
        self._is_loaded = True

    def show_dag(self):
        if self._is_loaded:
            plt.tight_layout()
            nx.draw_networkx(self._graph, arrows=True)
            plt.show()
            plt.clf()

    def get_sorted_nodes(self) -> List[Node]:
        if self._is_loaded:
            sorted_id_list = self.get_sorted_node_ids()
            sorted_node_list = []

            for node_id in sorted_id_list:
                sorted_node_list.append(self._nodes.get(node_id))

            return sorted_node_list

    def get_sorted_node_ids(self) -> List[str]:
        if self._is_loaded:
            topologically_ordered_list = list(nx.topological_sort(self._graph))

            return topologically_ordered_list

    def get_sub_pipelines(self):
        if not self._is_loaded:
            return None

        sub_pipelines = []
        sub_pipeline = []
        sub_pipeline_type = None

        sorted_node_list = self.get_sorted_nodes()

        for i in range(0, len(sorted_node_list)):
            node_type = sorted_node_list[i].node_type
            node = sorted_node_list[i]

            # il primo lo aggiungo a prescindere nella sottopipeline e setto il tipo
            if i == 0:
                sub_pipeline.append(node)
                sub_pipeline_type = node_type
                continue

            # Se il tipo del nodo è uguale a quello dell subpipeline lo aggiungo ad essa
            # Altrimenti aggiungo una tupla (tipo subpipeline, subpipeline) alla lista di subpipeline,
            # e resetto le variabili considerando il nuovo nodo.
            if node_type == sub_pipeline_type:
                sub_pipeline.append(node)
            else:
                sub_pipelines.append((sub_pipeline_type, sub_pipeline))
                sub_pipeline = [node]
                sub_pipeline_type = node_type

        sub_pipelines.append((sub_pipeline_type, sub_pipeline))

        return sub_pipelines


if __name__ == "__main__":
    sjp = DagCreator()

    sjp.create_dag("pandas_sklearn2.json")

    sjp.show_dag()

    pipeline = sjp.get_sub_pipelines()
