import importlib
import inspect
import re

from simple_repo.code_generator.imports_analyzer import extract_imports, ImportType
from simple_repo.code_generator.parents_analyzer import extract_parents


def get_all_classes(imported_module):
    return inspect.getmembers(imported_module, inspect.isclass)


def get_all_internal_classes(imported_module):
    return [
        m
        for m in get_all_classes(imported_module)
        if m[1].__module__ == imported_module.__name__
    ]


def get_all_internal_callables(imported_module):
    return [
        m
        for m in inspect.getmembers(imported_module, callable)
        if m[1].__module__ == imported_module.__name__
    ]


def get_class_from_imported_module(imported_module, classname):
    classes = get_all_classes(imported_module)

    class_list = [class_ for class_name, class_ in classes if class_name == classname]

    if not class_list:
        return None

    return class_list[0]


def get_code(class_):
    return inspect.getsource(class_)


def get_str_import_to_check(imp):
    if imp.has_alias():
        to_check = imp.alias
    else:
        to_check = imp.import_string
    return to_check


def check_import_usage(imp, cls):
    class_code = get_code(cls)

    to_check = get_str_import_to_check(imp)

    regex = r"(?:{}(?:\.\S+)?)".format(to_check)
    is_present = True if re.search(regex, class_code) is not None else False

    return is_present


def check_internal_callable_usage(callable_string, cls):
    class_code = get_code(cls)

    regex = r"(?:{}(?:\.\S+)?)".format(callable_string)
    is_present = True if re.search(regex, class_code) is not None else False

    return is_present


def extract_internal_callables(cls):
    calls = set()

    mod = inspect.getmodule(cls)
    module_callables = get_all_internal_callables(mod)

    for calname, call in module_callables:
        if calname != cls.__name__ and check_internal_callable_usage(calname, cls):
            calls.add(call)

    return calls


def get_callables(imp, cls):
    calls = set()

    module_name, package_name = get_package_module_name(imp)
    mod = importlib.import_module(module_name, package_name)
    attr = getattr(mod, imp.import_string)

    if inspect.ismodule(attr):
        code = get_code(cls)
        to_check = get_str_import_to_check(imp)
        matches = re.finditer(
            r"(?:{}\.(?P<str_name>[a-zA-Z.]+))".format(to_check), code, re.MULTILINE
        )
        for match in matches:
            m = match.group("str_name")
            clazz = getattr(attr, m)
            calls.add(clazz)
    else:
        calls.add(attr)

    return calls


def get_package_module_name(imp):
    full_name_parts = imp.from_string.split(
        "."
    )  # TODO gestire caso from simple_repo import Class
    if len(full_name_parts) == 1:
        package_name = ""
        module_name = imp.from_string
    else:
        package_name = ".".join(full_name_parts[:-1])
        module_name = "." + full_name_parts[-1]
    return module_name, package_name


def get_class_dependencies(cls: type):
    ext_imports = set()
    calls = set()

    all_imports = extract_imports(cls)

    for imp in all_imports:
        if check_import_usage(imp, cls):
            if imp.import_type == ImportType.EXTERNAL:
                ext_imports.add(imp)
            else:
                calls.update(get_callables(imp, cls))
        else:
            continue

    calls.update(extract_parents(cls))
    calls.update(extract_internal_callables(cls))

    return ext_imports, calls


if __name__ == "__main__":
    from simple_repo.simple_pandas.transform_nodes import (
        PandasColumnSelector,
        _filter_feature,
    )

    d = get_class_dependencies(_filter_feature)
    print("ok")
