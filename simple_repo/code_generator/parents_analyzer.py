def extract_parents(child_class):
    """
    Gets a set containing all the internal parents of the given class.

    With internal parents we refer to those parents that are defined
    in the library (not builtin or from external libraries).

    If the class does not have internal parents it returns an empty set.
    """
    parents = set()

    if hasattr(child_class, "bases") and child_class.__bases__:
        [
            parents.add(base)
            for base in child_class.__bases__
            if base.__module__.startswith("simple_repo")
        ]

    return parents
