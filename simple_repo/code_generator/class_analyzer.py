import inspect


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


def get_code(class_):
    return inspect.getsource(class_)
