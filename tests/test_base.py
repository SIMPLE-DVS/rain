import pytest

from rain.core.base import InputNode, OutputNode, ComputationalNode, SimpleNode
from rain.nodes import mongodb
from rain.nodes import pandas
from rain.nodes import sklearn
from rain.nodes import spark
import rain as sr


input_data = [
    spark.SparkCSVLoader,
    spark.SparkModelLoader,
    pandas.PandasCSVLoader,
    sklearn.IrisDatasetLoader,
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
    spark.SparkSaveModel,
    spark.SparkSaveDataset,
    pandas.PandasCSVWriter,
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
    sr.PandasGroupBy,
    sr.SimpleKMeans,
    sr.SklearnLinearSVC,
    sr.SparkPipelineNode,
    sr.Tokenizer,
    sr.HashingTF,
    sr.LogisticRegression,
    sr.SparkColumnSelector,
    sr.SparkSplitDataset,
    sr.ZScoreTrainer,
    sr.ZScorePredictor,
    sr.TPOTClassificationTrainer,
    sr.TPOTClassificationPredictor,
    sr.TPOTRegressionTrainer,
    sr.TPOTRegressionPredictor,
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
