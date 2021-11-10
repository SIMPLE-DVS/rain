import pytest

from simple_repo.simple_pandas.node_structure import PandasTransformer
import simple_repo.simple_pandas as sp
import simple_repo.simple_io.pandas_io as pio

pandas_nodes = [
    sp.PandasColumnsFiltering,
    sp.PandasPivot,
    sp.PandasRenameColumn,
    pio.PandasCSVLoader,
    pio.PandasCSVWriter,
    sp.PandasFilterRows,
    sp.PandasAddColumn,
    sp.PandasDropNan,
    # sp.PandasSelectRows,
    # sp.PandasReplaceColumn
]


@pytest.mark.parametrize("class_or_obj", pandas_nodes)
def test_pandas_methods(class_or_obj):
    """Checks whether the spark node has no method attributes."""
    assert (
        issubclass(class_or_obj, PandasTransformer)
        or issubclass(class_or_obj, pio.PandasInputNode)
        or issubclass(class_or_obj, pio.PandasOutputNode)
    ) and not hasattr(class_or_obj, "_methods")
