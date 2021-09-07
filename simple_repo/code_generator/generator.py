import inspect
from builtins import set
from dataclasses import dataclass
from queue import SimpleQueue, LifoQueue
import types
import re
from jinja2 import FileSystemLoader, Template, Environment
from enum import Enum
import networkx as nx
from matplotlib import pyplot as plt

from simple_repo.code_generator.base import ObjectInfo
from simple_repo.code_generator.class_analyzer import get_code
from simple_repo.code_generator.exceptions_analyzer import extract_exceptions
from simple_repo.code_generator.imports_analyzer import extract_imports
from simple_repo.code_generator.parameters_analyzer import extract_parameters
from simple_repo.code_generator.parents_analyzer import extract_parents
from simple_repo.simple_pandas.load_nodes import PandasCSVLoader
from simple_repo.simple_sklearn import svm
from simple_repo.simple_pandas import load_nodes, transform_nodes


def get_obj_info(obj_set, obj_name):
    for obj in obj_set:
        if obj.obj_clsname == obj_name:
            return obj

    return None


class InfoType(Enum):
    EXCEPTION = "exceptions"
    PARENT = "parents"
    CALLABLE = "callables"
    PARAMETER = "parameters"
    NODE = "nodes"


if __name__ == "__main__":

    def show_dag(dag):
        plt.tight_layout()
        nx.draw_networkx(dag, arrows=True)
        plt.show()
        plt.clf()

    nodes = set()
    queue = SimpleQueue()
    graph = nx.DiGraph()
    queue.put(ObjectInfo(PandasCSVLoader, InfoType.NODE))
    obj_infos = set()

    imports = set()

    while not queue.empty():
        edges = []
        new_nodes = set()

        info = queue.get()
        cls = info.obj_class
        clsname = cls.__name__

        # add imports
        [imports.add(imp) for imp in extract_imports(cls)]

        new_nodes.update(
            [ObjectInfo(exc, InfoType.EXCEPTION) for exc in extract_exceptions(cls)]
        )
        new_nodes.update(
            [ObjectInfo(param, InfoType.PARAMETER) for param in extract_parameters(cls)]
        )
        new_nodes.update(
            [ObjectInfo(par, InfoType.PARENT) for par in extract_parents(cls)]
        )

        [
            queue.put(new_obj)
            for new_obj in list(filter(lambda x: x not in nodes, new_nodes))
        ]

        edges.extend([(node.obj_clsname, clsname) for node in new_nodes])

        graph.add_edges_from(edges)

        nodes.update(new_nodes)
        nodes.add(info)

        # show_dag(graph)

    [node.retrieve_code() for node in nodes]

    print_order = list(nx.topological_sort(graph))

    print(print_order)

    nds = []

    for node_name in print_order:
        nds.append(get_obj_info(nodes, node_name))

    temp_file = FileSystemLoader("./")

    temp = Environment(line_statement_prefix="#", loader=temp_file).get_template(
        name="pipeline_template.py.jinja"
    )

    jinja_vars = {"imports": imports, "nodes": [node.code for node in nds]}

    # print(temp.render(parameters=parameters(), nodes_parents=[get_code(par) for par in pars]))

    temp.stream(**jinja_vars).dump("./pipeline.py")
