from typing import List, Tuple

from simple_repo.base import load_config, Node, get_class, Singleton
import networkx as nx
from matplotlib import pyplot as plt
import json

from simple_repo.simple_pandas.node_structure import PandasExecutor
from simple_repo.simple_sklearn.node_structure import SklearnExecutor


def get_nodes(config: dict, engine: str):
    nodes_types = {}
    nodes = {}
    edges = []

    for node in config.get(engine):
        node_id = node.get("node_id")
        json_node = Node(node_type=engine, **node)
        nodes_types[node_id] = engine
        nodes[node_id] = json_node

        if "then" in node.keys():
            for nxt in node.get("then"):
                edges.append((node_id, nxt.get("send_to")))

    return nodes_types, nodes, edges


def eseguitutto(subpipelines):
    nodes_insts = {}
    nodes_nexts = {}

    for node_tp, node_lst in subpipelines:
        for node in node_lst:
            node_class = get_class(node.node)
            if node_tp == "sklearn":
                node_inst = node_class(node.execute, **node.parameters)
            else:
                node_inst = node_class(**node.parameters)

            nodes_insts[node.node_id] = node_inst

            if node.then:
                nodes_nexts[node.node_id] = node.then

    for node_name, node_inst in nodes_insts.items():
        node_inst.execute()

        if node_name not in nodes_nexts.keys():
            continue

        for nxt in nodes_nexts.get(node_name):
            receiver_id = nxt.get("send_to")

            receiver = nodes_insts.get(receiver_id)

            for k, v in nxt.items():
                if k == "send_to":
                    continue

                actual_node_output = node_inst.get_output_value(k)
                receiver.set_input_value(v, actual_node_output)


class SimplePipeline:
    executors = {"pandas": PandasExecutor(), "sklearn": SklearnExecutor()}

    def __init__(self, pipeline: List[Tuple[str, Node]]):
        self._pipeline = pipeline

    def execute_pipeline(self):
        for subpipeline in self._pipeline:
            self.executors.get(subpipeline[0]).execute(subpipeline[1])


class SimpleJSONParser:
    """
    Takes a json and parse it to a DAG.
    """

    engines = ["pandas", "sklearn", "spark"]

    def __init__(self):
        self._graph = nx.DiGraph()
        self._is_loaded = False
        self._nodes_types = {}
        self._nodes = {}
        self._sorted_nodes = None

    def parse_configuration(self, config_file_path: str):

        config = load_config(config_file_path)

        edges = []

        for tp in self.engines:
            if tp in config.keys():
                nodes_types_app, nodes_app, edges_app = get_nodes(config, tp)

                self._nodes_types.update(nodes_types_app)
                self._nodes.update(nodes_app)
                edges.extend(edges_app)

        self._graph.add_edges_from(edges)

        self._is_loaded = True

    def show_dag(self):
        if self._is_loaded:
            plt.tight_layout()
            nx.draw_networkx(self._graph, arrows=True)
            plt.show()
            plt.clf()

    def get_sorted_nodes(self):
        if self._is_loaded:
            sorted_id_list = self.get_sorted_node_ids()
            sorted_node_list = []

            for node_id in sorted_id_list:
                sorted_node_list.append(self._nodes.get(node_id))

            return sorted_node_list

    def get_sorted_node_ids(self):
        if self._is_loaded:
            topologically_ordered_list = list(nx.topological_sort(self._graph))

            return topologically_ordered_list

    def get_subpipelines(self):
        if not self._is_loaded:
            return None

        subpipelines = []
        subpipeline = []
        subpipeline_type = None

        sorted_node_list = self.get_sorted_node_ids()

        for i in range(0, len(sorted_node_list)):
            node_type = self._nodes_types.get(sorted_node_list[i])
            node = self._nodes.get(sorted_node_list[i])

            # il primo lo aggiungo a prescindere nella sottopipeline e setto il tipo
            if i == 0:
                subpipeline.append(node)
                subpipeline_type = node_type
                continue

            # Se il tipo del nodo è uguale a quello dell subpipeline lo aggiungo ad essa
            # Altrimenti aggiungo una tupla (tipo subpipeline, subpipeline) alla lista di subpipeline,
            # e resetto le variabili considerando il nuovo nodo.
            if node_type == subpipeline_type:
                subpipeline.append(node)
            else:
                subpipelines.append((subpipeline_type, subpipeline))
                subpipeline = [node]
                subpipeline_type = node_type

        subpipelines.append((subpipeline_type, subpipeline))

        return subpipelines

    # dato che il dag viene creato leggendo solamente gli id ed i then,
    # aggiungere un controllo che si assicuri che tutti gli id presenti
    # nel dag siano effettivamente dei nodi. Infatti potrei scrivere nel
    # then di un nodo un id che poi non è mai presente nel json.


if __name__ == "__main__":
    sjp = SimpleJSONParser()

    sjp.parse_configuration("pandas_sklearn2.json")

    sjp.show_dag()

    pipeline = sjp.get_subpipelines()

    # eseguitutto(subpipelines)

    main_exec = SimplePipeline(pipeline)
