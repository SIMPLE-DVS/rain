import inspect
from dataclasses import dataclass
from queue import SimpleQueue, LifoQueue
import types
import re
from jinja2 import FileSystemLoader, Template, Environment

from simple_repo.simple_sklearn import svm
from simple_repo.simple_pandas import load_nodes, transform_nodes


@dataclass
class ImportInfo:
    import_string: str
    from_string: str = ""
    alias: str = ""

    def __str__(self):
        return "{}{}{}".format(
            "from {} ".format(self.from_string) if self.from_string != "" else "",
            "import {}".format(self.import_string),
            " as {}".format(self.alias) if self.alias != "" else "",
        )


@dataclass
class ObjectInfo:
    obj_class = None  # class
    imports = []  # ImportInfos list
    exceptions = []  # exceptions lists
    parameters = []  # parameters list
    code = ""
    parents = []  # nodesinfo list


def get_all_classes(imported_module):
    return inspect.getmembers(imported_module, inspect.isclass)


def get_all_internal_classes(imported_module):
    return [
        m
        for m in get_all_classes(imported_module)
        if m[1].__module__ == imported_module.__name__
    ]


def get_class_from_imported_module(imported_module, classname):
    classes = get_all_classes(imported_module)

    class_list = [class_ for class_name, class_ in classes if class_name == classname]

    if not class_list:
        return None

    return class_list[0]


def create_import_info(import_string):  # noqa W605
    """
    https://regex101.com/r/xFtey5/1 per provarla

    la regex qua sotto matcha tutti le stringhe sottostanti
    ed estrapola i contenuti di from, import e as.

    from abc.lmn import pqr
    from abc.lmn import pqr as xyz
    import abc
    import abc, as, xyz
    from abc.lmn import pqr, lmn

    Non matcha un pattern del tipo:

    from time import process_time as tic, process_time as toc

    non capisco perché debba essere usato un import del genere quindi per ora va bene così.
    """
    match_from = re.match(
        r"(?m)^(?:from[ ]+(?P<from_string>\S+)[ ]+)?import[ ]+(?P<import_string>\S+(?:, \S+)*?)(?:[ ]+as[ ]+("
        r"?P<alias>\S+))?[ ]*$",
        import_string,
    )
    from_string = (
        match_from.group("from_string") if match_from.group("from_string") else ""
    )
    import_string = match_from.group("import_string")
    alias = match_from.group("alias") if match_from.group("alias") else ""

    return ImportInfo(import_string, from_string=from_string, alias=alias)


def get_module_imports(mod):
    """
    La regex qua sotto estrapola tutti gli import che iniziano o meno con from da
    un modulo
    """
    source = inspect.getsource(mod)
    imports = []
    for fr in re.finditer(r"(?m)^(?:from[ ]+(\S+)[ ]+)?import[ ]+(.*)$", source):
        if fr is not None:
            imports.append(fr.group(0))

    return imports


def get_class_exceptions(node_class):
    """
    La regex qua sotto
    estrapola tutte le eccezioni che utilizzate all'interno della classe.

    Tutte quelle che non sono custom, e quindi appartenenti al modulo
    simple_repo.exception vengono escluse perché dovrebbero essere built-in.

    TODO: Rendere il meccanismo più intelligente, non è detto che una
    eccezione custom sia definita solamente nel modulo exception.
    """
    source = get_code(node_class)
    exceptions = []
    for fr in re.finditer(r"(?m)^(?:.*raise (?P<exception>\S+)\()$", source):
        if fr is not None:
            exceptions.append(fr.group("exception"))

    return exceptions


def get_code(class_):
    return inspect.getsource(class_)


def get_parameters_classes():
    from simple_repo import parameter

    classes = get_all_internal_classes(parameter)

    classes = [c for _, c in classes]

    return classes


def parameters():
    classes = get_parameters_classes()
    for class_ in classes:
        yield get_code(class_)


def get_all_parents_classes(child_class):
    to_eval = SimpleQueue()
    [to_eval.put(parent) for parent in child_class.__bases__]
    parents = list()

    while not to_eval.empty():
        curr_class = to_eval.get()
        [to_eval.put(parent) for parent in curr_class.__bases__ if parent is not object]
        parents.append(curr_class)

    parents.reverse()

    return parents


def parents(child):
    classes = get_all_parents_classes(child)
    for parent in classes:
        yield inspect.getsource(parent)


def concrs():
    concretes = [
        load_nodes.PandasCSVLoader,
        svm.SklearnLinearSVC,
        transform_nodes.PandasAddColumn,
        load_nodes.PandasCSVWriter,
    ]
    for cr in concretes:
        yield cr


def extract_class_info(node_class):
    info = ObjectInfo()
    info.obj_class = classs
    info.code = get_code(classs)
    info.parameters = get_parameters_classes()
    info.parents = get_all_parents_classes(classs)
    imps = get_module_imports(load_nodes)
    info.imports = [create_import_info(imp) for imp in imps if "simple_repo" not in imp]
    info.exceptions = get_class_exceptions(classs)

    print(info.exceptions)

    return info


if __name__ == "__main__":
    from simple_repo import base

    classs = get_class_from_imported_module(base, "SimpleNode")

    class_info = extract_class_info(classs)

    # classs = get_class_from_imported_module(node_structure, "PandasNode")
    #
    # code = inspect.getsource(classs)
    #
    temp_file = FileSystemLoader("./")

    temp = Environment(line_statement_prefix="#", loader=temp_file).get_template(
        name="pipeline_template.py.jinja"
    )

    pars = list()

    pares = [par for par in class_info.parents if par is not object]
    for par in pares:
        if par not in pars:
            pars.append(par)

    jinja_vars = {
        "imports": class_info.imports,
        "parameters": parameters(),
        "nodes_parents": [get_code(par) for par in pars],
        "concrete_node": class_info.code,
    }

    # print(temp.render(parameters=parameters(), nodes_parents=[get_code(par) for par in pars]))

    temp.stream(**jinja_vars).dump("./pipeline.py")
