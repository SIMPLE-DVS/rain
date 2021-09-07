import inspect
from queue import SimpleQueue


def extract_parents(child_class, excluding=[]):
    """
    Gets all the parents of the given class excluding the ones specified.
    """
    to_eval = SimpleQueue()
    [
        to_eval.put(parent)
        for parent in child_class.__bases__
        if parent is not object and parent not in excluding
    ]
    parents = list()

    while not to_eval.empty():
        curr_class = to_eval.get()
        [
            to_eval.put(parent)
            for parent in curr_class.__bases__
            if parent is not object
            and parent not in excluding
            and parent.__module__.startswith("simple_repo")
        ]
        if curr_class.__module__.startswith("simple_repo"):
            parents.append(curr_class)

    parents.reverse()

    return parents
