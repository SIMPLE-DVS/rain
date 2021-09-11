from builtins import set
from queue import SimpleQueue
from jinja2 import FileSystemLoader, Environment
from enum import Enum
import networkx as nx
from matplotlib import pyplot as plt

from simple_repo.code_generator.class_analyzer import get_code, get_class_dependencies


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


def write_file(imp, cal, graph):
    print_order = list(nx.topological_sort(graph))

    print(print_order)

    temp_file = FileSystemLoader("./")
    # line_statement_prefix="#",
    temp = Environment(loader=temp_file).get_template(name="pipeline_template.py.jinja")

    jinja_vars = {"imports": imp, "nodes": code_generator(cal, print_order)}

    # print(temp.render(parameters=parameters(), nodes_parents=[get_code(par) for par in pars]))

    temp.stream(**jinja_vars).dump("./pipeline.py")


def show_dag(dag):
    plt.tight_layout()
    nx.draw_networkx(dag, arrows=True)
    plt.show()
    plt.clf()


if __name__ == "__main__":
    from simple_repo.simple_pandas.load_nodes import PandasCSVLoader
    from simple_repo.simple_pandas.transform_nodes import PandasColumnSelector

    imp, cal, graph = generate_code([PandasCSVLoader, PandasColumnSelector])

    # show_dag(graph)

    write_file(imp, cal, graph)

    print("ok")
