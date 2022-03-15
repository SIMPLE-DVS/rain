import pytest

from simple_repo.base import InputNode, OutputNode, ComputationalNode, SimpleNode
from simple_repo import mongodb
from simple_repo import simple_pandas
from simple_repo import simple_sklearn
from simple_repo import simple_spark
import simple_repo as sr


input_data = [
    simple_spark.SparkCSVLoader,
    simple_spark.SparkModelLoader,
    simple_pandas.PandasCSVLoader,
    simple_sklearn.IrisDatasetLoader,
    mongodb.MongoCSVReader,
]


@pytest.mark.parametrize("class_or_obj", input_data)
def test_input_node_attributes(class_or_obj):
    """Checks whether the input node class or object has all the needed class attributes."""
    assert (
        issubclass(class_or_obj, InputNode)
        and hasattr(class_or_obj, "_output_vars")
        and not hasattr(class_or_obj, "_input_vars")
        and not hasattr(class_or_obj, "_methods")
    )


output_data = [
    simple_spark.SparkSaveModel,
    simple_spark.SparkSaveDataset,
    simple_pandas.PandasCSVWriter,
    mongodb.MongoCSVWriter,
]


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
    sr.PandasColumnsFiltering,
    sr.PandasPivot,
    sr.PandasRenameColumn,
    sr.PandasReplaceColumn,
    sr.PandasAddColumn,
    sr.PandasDropNan,
    sr.PandasFilterRows,
    sr.PandasSelectRows,
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
