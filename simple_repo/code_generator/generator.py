import os
from builtins import set
from queue import SimpleQueue
from jinja2 import FileSystemLoader, Environment
from jinja2.ext import debug
from enum import Enum
import networkx as nx
from matplotlib import pyplot as plt

from simple_repo.base import get_class
from simple_repo.code_generator.class_analyzer import (
    get_code,
    get_class_dependencies,
    get_all_internal_callables,
)
from simple_repo.dag import SimpleJSONParser


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


def generate_code(classes: list):
    graph = nx.DiGraph()
    imports = set()
    callables = set()

    queue = SimpleQueue()

    for cls in classes:
        queue.put(cls)

    while not queue.empty():
        cls = queue.get()

        cls_imports, cls_callables = get_class_dependencies(cls)

        difference = cls_callables.difference(callables)

        for call in difference:
            queue.put(call)

        edges = [(call.__name__, cls.__name__) for call in cls_callables]

        graph.add_edges_from(edges)

        imports.update(cls_imports)
        callables.update(cls_callables)
        callables.add(cls)

    return imports, {cal.__name__: cal for cal in callables}, graph


def code_generator(callables: dict, ordered_cal_names: list):
    for cal_name in ordered_cal_names:
        yield get_code(callables.get(cal_name))


def generate_executor_callables():
    from simple_repo import execution

    callables = get_all_internal_callables(execution)

    return callables


def get_nodes_instantiation_str(node_list):
    return [node.node_instantiation_str() for node in node_list]


def generate_subpipelines_code(subpipelines):
    for i, subpip in enumerate(subpipelines):
        node_list_str = get_nodes_instantiation_str(subpip[1])
        yield i, subpip[0], node_list_str


def write_file(imp, cal, graph, subpipelines):
    print_order = list(nx.topological_sort(graph))

    print(print_order)

    temp_file = FileSystemLoader("./")
    # line_statement_prefix="#",
    temp = Environment(loader=temp_file).get_template(
        name="./code_generator/pipeline_template.py.jinja"
    )

    jinja_vars = {
        "imports": imp,
        "nodes": code_generator(cal, print_order),
        "subpipelines": subpipelines,
        "subpipelines_len": len(subpipelines),
    }

    # print(temp.render(parameters=parameters(), nodes_parents=[get_code(par) for par in pars]))

    temp.stream(**jinja_vars).dump("./code_generator/pipeline.py")

    os.system("black ./code_generator/pipeline.py")


def show_dag(dag):
    plt.tight_layout()
    nx.draw_networkx(dag, arrows=True)
    plt.show()
    plt.clf()


if __name__ == "__main__":
    from simple_repo.simple_pandas.load_nodes import PandasCSVLoader
    from simple_repo.simple_pandas.transform_nodes import PandasColumnSelector

    sjp = SimpleJSONParser()

    sjp.parse_configuration("./pandas_sklearn.yaml")

    # sjp.show_dag()

    subpipelines = sjp.get_sub_pipelines()

    classes = []

    for _, node_list in subpipelines:
        classes.extend(map(lambda x: get_class(x.node), node_list))

    calls = [cal for _, cal in generate_executor_callables()] + classes

    imp, cal, graph = generate_code(calls)

    # show_dag(graph)

    write_file(imp, cal, graph, subpipelines)

    print([cd for cd in get_nodes_instantiation_str(subpipelines[0][1])])
