import importlib
import inspect
import re
from dataclasses import dataclass


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

    def __hash__(self) -> int:
        return super().__hash__()


def create_import_info(import_string):  # noqa W605
    """
    https://regex101.com/r/xFtey5/1 per provarla

    la regex qua sotto matcha tutti le stringhe sottostanti
    ed estrapola i contenuti di from, import e as.

    from abc.lmn import pqr
    from abc.lmn import pqr as xyz
    import abc
    import abc as xyz
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


def extract_imports(obj_class):
    mod = obj_class.__module__
    mod = importlib.import_module(mod)
    imps = get_module_imports(mod)
    return [create_import_info(imp) for imp in imps]
