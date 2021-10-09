from pyspark.sql import DataFrame
import pytest
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from sklearn.datasets import load_iris

import simple_repo as sr
from simple_repo.simple_spark.node_structure import SparkNode, SparkNodeSession
from pyspark.ml.feature import Tokenizer


computational_spark_nodes = [
    sr.SparkPipelineNode,
    sr.Tokenizer,
    sr.HashingTF,
    sr.LogisticRegression,
    sr.SparkColumnSelector,
    sr.SparkSplitDataset,
]

computational_spark_nodes_instance = [
    sr.SparkPipelineNode([]),
    sr.Tokenizer("", ""),
    sr.HashingTF("", ""),
    sr.LogisticRegression(2, 3),
    # sr.SparkColumnSelector(features=[{"col":""}]),
    sr.SparkSplitDataset(1, 2),
]

spark_nodes = [
    sr.SparkCSVLoader,
    sr.SparkModelLoader,
    sr.SaveModel,
    sr.SaveDataset,
    sr.SparkPipelineNode,
    sr.Tokenizer,
    sr.HashingTF,
    sr.LogisticRegression,
    sr.SparkColumnSelector,
    sr.SparkSplitDataset,
]


@pytest.mark.parametrize("class_or_obj", computational_spark_nodes)
def test_spark_methods(class_or_obj):
    """Checks whether the spark node has no method attributes."""
    assert issubclass(class_or_obj, SparkNode) and not hasattr(class_or_obj, "_methods")


@pytest.mark.parametrize("class_or_obj", computational_spark_nodes_instance)
def test_spark_computational_instance(class_or_obj):
    """Checks whether the spark computational node has the computational instance."""
    assert isinstance(class_or_obj, SparkNode) and hasattr(
        class_or_obj, "computational_instance"
    )


@pytest.mark.parametrize("class_or_obj", spark_nodes)
def test_spark_instance(class_or_obj):
    """Checks whether the spark node has the spark session attribute."""
    assert issubclass(class_or_obj, SparkNodeSession) and hasattr(class_or_obj, "spark")


@pytest.fixture
def iris_data():
    yield load_iris(as_frame=True).data


class TestSparkCsvLoader:
    def test_dataset_load(self, tmpdir, iris_data):
        iris = SparkSession.builder.getOrCreate().createDataFrame(iris_data).toPandas()
        tmpcsv = tmpdir / "tmp_iris.csv"
        iris.to_csv(tmpcsv, index=False)
        loader = sr.SparkCSVLoader(path=tmpcsv.__str__(), header=True)
        loader.execute()
        iris_loaded = loader.dataset
        assert iris.shape == iris_loaded.toPandas().shape
        assert isinstance(iris_loaded, DataFrame)


class TestSparkModelLoader:
    def test_model_load(self, tmpdir):
        model = PipelineModel([Tokenizer()])
        tmpmod = tmpdir / "tmp_model.pkl"
        model.write().overwrite().save(tmpmod.__str__())
        loader = sr.SparkModelLoader(path=tmpmod.__str__())
        loader.execute()
        assert isinstance(loader.model, PipelineModel)
        assert isinstance(loader.model.stages[0], Tokenizer)


class TestTestSaveModel:
    pass


class TestSaveDataset:
    pass


class TestTokenizer:
    pass


class TestHashingTf:
    pass


class TestLogisticRegression:
    pass


class TestSparkPipeline:
    pass
