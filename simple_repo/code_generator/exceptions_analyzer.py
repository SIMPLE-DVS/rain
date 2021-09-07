import re

from simple_repo.code_generator.class_analyzer import get_code, get_all_internal_classes


def get_exceptions_as_strings(obj_class):
    """
    La regex qua sotto
    estrapola tutte le eccezioni che utilizzate all'interno della classe.

    Tutte quelle che non sono custom, e quindi appartenenti al modulo
    simple_repo.exception vengono escluse perché dovrebbero essere built-in.

    TODO: Rendere il meccanismo più intelligente, non è detto che una
    eccezione custom sia definita solamente nel modulo exception.
    """
    source = get_code(obj_class)
    exceptions = []
    for fr in re.finditer(r"(?m)^(?:.*raise (?P<exception>\S+)\()$", source):
        if fr is not None:
            exceptions.append(fr.group("exception"))

    return exceptions


def extract_exceptions(obj_class):
    import simple_repo.exception as ex

    exceptions = get_all_internal_classes(ex)
    class_exceptions = get_exceptions_as_strings(obj_class)

    to_return = [cls for clsname, cls in exceptions if clsname in class_exceptions]

    return to_return
