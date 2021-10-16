import pytest

import simple_repo.simple_sklearn as ss
from simple_repo.simple_sklearn.node_structure import SklearnNode

sklearn_nodes = [ss.SklearnLinearSVC, ss.SimpleKMeans]


@pytest.mark.parametrize("class_or_obj", sklearn_nodes)
def test_sklearn_methods(class_or_obj):
    """Checks whether the spark node has no method attributes."""
    assert issubclass(class_or_obj, SklearnNode) and hasattr(class_or_obj, "_methods")


@pytest.mark.parametrize("class_or_obj", sklearn_nodes)
def test_sklearn_execute(class_or_obj):
    """Checks whether the spark node has no method attributes."""
    with pytest.raises(Exception):
        class_or_obj(execute=["non_existing_method"])
