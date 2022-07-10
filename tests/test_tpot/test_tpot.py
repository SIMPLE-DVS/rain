import pytest
from sklearn.datasets import load_iris
import rain as sr


@pytest.fixture
def iris():
    yield load_iris(as_frame=True)


def test_tpot_classification(iris):
    tpot_classification_trainer = sr.TPOTClassificationTrainer("tct", target_feature="target", export_script=False,
                                                               generations=2, population_size=2, cv=2, verbosity=0)
    tpot_classification_trainer.dataset = iris.frame
    tpot_classification_trainer.execute()
    model = tpot_classification_trainer.model
    assert model is not None
    tpot_classification_predictor = sr.TPOTClassificationPredictor("tcp")
    tpot_classification_predictor.dataset = iris.data
    tpot_classification_predictor.model = model
    tpot_classification_predictor.execute()
    assert tpot_classification_predictor.predictions is not None


def test_tpot_regression(iris):
    tpot_regression_trainer = sr.TPOTRegressionTrainer("trt", target_feature="sepal length (cm)", export_script=False,
                                                       generations=2, population_size=2, cv=2, verbosity=0)
    tpot_regression_trainer.dataset = iris.frame
    tpot_regression_trainer.execute()
    model = tpot_regression_trainer.model
    assert model is not None
    tpot_regression_predictor = sr.TPOTRegressionPredictor("trp")
    tpot_regression_predictor.dataset = iris.data
    tpot_regression_predictor.model = model
    tpot_regression_predictor.execute()
    assert tpot_regression_predictor.predictions is not None
