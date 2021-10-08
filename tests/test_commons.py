import pytest

from simple_repo.exception import ParameterNotFound
from simple_repo.simple_sklearn.node_structure import (
    SklearnClassifier,
    SklearnEstimator,
    SklearnClusterer,
)

diomadonna = [
    (SklearnEstimator, ["fit_dataset"], ["fitted_model"], ["fit"]),
    (
        SklearnClassifier,
        ["fit_dataset", "predict_dataset", "score_dataset"],
        ["fitted_model", "predictions", "scores"],
        ["fit", "predict", "score"],
    ),
    (
        SklearnClusterer,
        ["fit_dataset", "predict_dataset", "score_dataset", "transform_dataset"],
        ["fitted_model", "predictions", "scores", "transformed_dataset"],
        ["fit", "predict", "score", "transform"],
    ),
]


@pytest.mark.parametrize("class_, in_vars, out_vars, methods_vars", diomadonna)
def test_class_integrity(class_, in_vars, out_vars, methods_vars):
    if hasattr(class_, "_input_vars") and in_vars is not None:
        assert set(in_vars) == set(class_._input_vars.keys())
    if hasattr(class_, "_output_vars") and out_vars is not None:
        assert set(out_vars) == set(class_._output_vars.keys())
    if hasattr(class_, "_methods") and methods_vars is not None:
        assert set(methods_vars) == set(class_._methods.keys())


def check_param_not_found(class_, **kwargs):
    """Checks whether the class raises a TypeError exception when instantiating,
    meaning that an invalid argument has been passed."""
    with pytest.raises(TypeError):
        class_(**kwargs)
