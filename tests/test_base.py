import pytest

from simple_repo.base import InputNode, OutputNode, ComputationalNode, SimpleNode
from simple_repo import simple_io as sio
import simple_repo as sr


input_data = [sio.SparkCSVLoader, sio.SparkModelLoader, sio.PandasCSVLoader]


@pytest.mark.parametrize("class_or_obj", input_data)
def test_input_node_attributes(class_or_obj):
    """Checks whether the input node class or object has all the needed class attributes."""
    assert (
        issubclass(class_or_obj, InputNode)
        and hasattr(class_or_obj, "_output_vars")
        and not hasattr(class_or_obj, "_input_vars")
        and not hasattr(class_or_obj, "_methods")
    )


output_data = [sio.SaveModel, sio.SaveDataset, sio.PandasCSVWriter]


@pytest.mark.parametrize("class_or_obj", output_data)
def test_output_node_attributes(class_or_obj):
    """Checks whether the input node class or object has all the needed class attributes."""
    assert (
        issubclass(class_or_obj, OutputNode)
        and hasattr(class_or_obj, "_input_vars")
        and not hasattr(class_or_obj, "_output_vars")
        and not hasattr(class_or_obj, "_methods")
    )


computational_data = [
    sr.PandasColumnSelector,
    sr.PandasPivot,
    sr.PandasRenameColumn,
    sr.SimpleKMeans,
    sr.SklearnLinearSVC,
    sr.SparkPipelineNode,
    sr.Tokenizer,
    sr.HashingTF,
    sr.LogisticRegression,
    sr.SparkColumnSelector,
    sr.SparkSplitDataset,
]


@pytest.mark.parametrize("class_or_obj", computational_data)
def test_computational_node_attributes(class_or_obj):
    """Checks whether the input node class or object has all the needed class attributes."""
    assert (
        issubclass(class_or_obj, ComputationalNode)
        and hasattr(class_or_obj, "_input_vars")
        and hasattr(class_or_obj, "_output_vars")
    )


empty_data = [SimpleNode]


@pytest.mark.parametrize("class_or_obj", empty_data)
def test_empty_node(class_or_obj):
    assert (
        issubclass(class_or_obj, SimpleNode)
        and not hasattr(class_or_obj, "_input_vars")
        and not hasattr(class_or_obj, "_output_vars")
        and not hasattr(class_or_obj, "_methods")
    )
