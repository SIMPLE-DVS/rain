from functools import reduce

from simple_repo.code_generator.class_analyzer import get_all_internal_classes


def get_parameters_classes():
    from simple_repo import parameter

    """
    Gets all the parameters classes
    """

    classes = get_all_internal_classes(parameter)

    classes = [c for _, c in classes]

    return classes


def get_parameters_classes(excluding=[]):  # noqa F811
    """
    Gets all the parameters classes excluding the ones specified.
    """
    from simple_repo import parameter

    classes = get_all_internal_classes(parameter)

    classes = [c for _, c in classes if c not in excluding]

    return classes


def extract_parameters(obj_class):
    """
    Gets all the parameters classes associated to the given class
    excluding the ones specified.
    """

    def class_accumulator(class_list, cls):
        if cls not in class_list:
            class_list.append(cls)

        return class_list

    if not hasattr(obj_class, "_parameters"):
        return []

    pars = map(lambda x: x.__class__, obj_class._parameters.values())

    pars_classes = reduce(class_accumulator, pars, [])

    return pars_classes
