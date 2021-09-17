import os
from builtins import set
from queue import SimpleQueue
from jinja2 import FileSystemLoader, Environment
from enum import Enum
import networkx as nx
from matplotlib import pyplot as plt

from simple_repo.base import get_class
from simple_repo.code_generator.class_analyzer import (
    get_code,
    get_class_dependencies,
    get_all_internal_callables,
)
from simple_repo.dag import DagCreator


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


def generate_code(class_list: list):
    di_graph = nx.DiGraph()
    imports = set()
    callables = set()

    queue = SimpleQueue()

    for cls in class_list:
        queue.put(cls)

    while not queue.empty():
        cls = queue.get()

        cls_imports, cls_callables = get_class_dependencies(cls)

        difference = cls_callables.difference(callables)

        for call in difference:
            queue.put(call)

        edges = [(call.__name__, cls.__name__) for call in cls_callables]

        di_graph.add_edges_from(edges)

        imports.update(cls_imports)
        callables.update(cls_callables)
        callables.add(cls)

    return imports, {cal.__name__: cal for cal in callables}, di_graph


def code_generator(callables: dict, ordered_cal_names: list):
    for cal_name in ordered_cal_names:
        yield get_code(callables.get(cal_name))


def generate_executor_callables():
    from simple_repo import execution

    callables = get_all_internal_callables(execution)

    return callables


def get_nodes_instantiation_str(nodes):
    return [node.node_instantiation_str() for node in nodes]


def generate_sub_pipelines_code(sub_pipelines):
    for i, sub_pip in enumerate(sub_pipelines):
        node_list_str = get_nodes_instantiation_str(sub_pip[1])
        yield i, sub_pip[0], node_list_str


def write_file(imports, callables, graph, sub_pipelines):
    print_order = list(nx.topological_sort(graph))

    print(print_order)

    temp_file = FileSystemLoader("./")
    # line_statement_prefix="#",
    temp = Environment(loader=temp_file).get_template(
        name="./code_generator/pipeline_template.py.jinja"
    )

    jinja_vars = {
        "imports": imports,
        "nodes": code_generator(callables, print_order),
        "subpipelines": sub_pipelines,
        "subpipelines_len": len(sub_pipelines),
    }

    # print(temp.render(parameters=parameters(), nodes_parents=[get_code(par) for par in pars]))

    temp_stream = temp.stream(**jinja_vars)

    return temp_stream

    # temp_stream.dump("./code_generator/pipeline.py")

    # os.system("black ./code_generator/pipeline.py")


def show_dag(dag):
    plt.tight_layout()
    nx.draw_networkx(dag, arrows=True)
    plt.show()
    plt.clf()


class ScriptGenerator:
    def __init__(self, pipeline: list):
        super(ScriptGenerator, self).__init__()
        self.pipeline = pipeline
        self.classes = []

    def generate_script(self):
        for _, nodes in self.pipeline:
            self.classes.extend(map(lambda x: get_class(x.node), nodes))

        calls = [cal for _, cal in generate_executor_callables()] + self.classes
        imports, callables, graph = generate_code(calls)
        script = write_file(imports, callables, graph, self.pipeline)
        return script


if __name__ == "__main__":

    sjp = DagCreator()

    sjp.create_dag("./pandas_sklearn.yaml")

    # sjp.show_dag()

    pipe = sjp.get_sub_pipelines()

    classes = []

    for _, node_list in pipe:
        classes.extend(map(lambda x: get_class(x.node), node_list))

    cl = [cal for _, cal in generate_executor_callables()] + classes

    imp, cal, gr = generate_code(cl)

    # show_dag(graph)

    write_file(imp, cal, gr, pipe)

    print([cd for cd in get_nodes_instantiation_str(pipe[0][1])])
