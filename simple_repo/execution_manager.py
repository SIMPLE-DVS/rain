from base import load_config, Node
import networkx as nx
from matplotlib import pyplot as plt
import json

if __name__ == "__main__":

    graph = nx.DiGraph()

    pd_config = load_config("pandas_sklearn.json")

    nodes_types = {}
    nodes = {}
    edges = []

    for node in pd_config.get("pandas"):
        print(json.dumps(node, indent=4))
        node_id = node.get("node_id")
        json_node = Node(node_type="pandas", **node)
        nodes_types[node_id] = "pandas"
        nodes[node_id] = json_node

        try:
            for next in node.get("then"):
                edges.append((node_id, next.get("send_to")))
        except Exception:
            pass

    for node in pd_config.get("sklearn"):
        print(json.dumps(node, indent=4))
        node_id = node.get("node_id")
        json_node = Node(node_type="sklearn", **node)
        nodes_types[node_id] = "sklearn"
        nodes[node_id] = json_node

        try:
            for next in node.get("then"):
                edges.append((node_id, next.get("send_to")))
        except Exception:
            pass

    graph.add_edges_from(edges)

    plt.tight_layout()
    nx.draw_networkx(graph, arrows=True)
    plt.show()
    plt.clf()

    topologically_ordered_list = list(nx.topological_sort(graph))

    print(topologically_ordered_list)

    # dato che il dag viene creato leggendo solamente gli id ed i then,
    # aggiungere un controllo che si assicuri che tutti gli id presenti
    # nel dag siano effettivamente dei nodi. Infatti potrei scrivere nel
    # then di un nodo un id che poi non è mai presente nel json.

    subpipelines = []
    subpipeline = []
    subpipeline_type = None

    for i in range(0, len(topologically_ordered_list)):
        node_type = nodes_types.get(topologically_ordered_list[i])
        node = nodes.get(topologically_ordered_list[i])

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

    print(subpipelines)
