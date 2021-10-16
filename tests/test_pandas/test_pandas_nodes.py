import pytest

from simple_repo.simple_pandas.node_structure import PandasNode
import simple_repo.simple_pandas as sp
import simple_repo.simple_io.pandas_io as pio

pandas_nodes = [
    sp.PandasColumnSelector,
    sp.PandasPivot,
    sp.PandasRenameColumn,
    pio.PandasCSVLoader,
    pio.PandasCSVWriter,
]


@pytest.mark.parametrize("class_or_obj", pandas_nodes)
def test_pandas_methods(class_or_obj):
    """Checks whether the spark node has no method attributes."""
    assert (
        issubclass(class_or_obj, PandasNode)
        or issubclass(class_or_obj, pio.PandasInputNode)
        or issubclass(class_or_obj, pio.PandasOutputNode)
    ) and not hasattr(class_or_obj, "_methods")
