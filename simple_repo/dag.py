from simple_repo.base import load_config, Node, load_yaml_config
import networkx as nx
from matplotlib import pyplot as plt


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

        # config = load_config(config_file_path)
        config = load_yaml_config(config_file_path)

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

    def get_sub_pipelines(self):
        if not self._is_loaded:
            return None

        sub_pipelines = []
        sub_pipeline = []
        sub_pipeline_type = None

        sorted_node_list = self.get_sorted_node_ids()

        for i in range(0, len(sorted_node_list)):
            node_type = self._nodes_types.get(sorted_node_list[i])
            node = self._nodes.get(sorted_node_list[i])

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

    # dato che il dag viene creato leggendo solamente gli id ed i then,
    # aggiungere un controllo che si assicuri che tutti gli id presenti
    # nel dag siano effettivamente dei nodi. Infatti potrei scrivere nel
    # then di un nodo un id che poi non è mai presente nel json.


if __name__ == "__main__":
    sjp = SimpleJSONParser()

    sjp.parse_configuration("pandas_sklearn2.json")

    sjp.show_dag()

    pipeline = sjp.get_sub_pipelines()
