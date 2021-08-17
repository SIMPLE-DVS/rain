from sklearn.svm import LinearSVC

from simple_repo.parameter import KeyValueParameter
from simple_repo.simple_sklearn.node_structure import SklearnClassifier


class SklearnLinearSVC(SklearnClassifier):
    _parameters = {
        "penalty": KeyValueParameter("penalty", str, value="l2"),
        "loss": KeyValueParameter("loss", str, value="squared_hinge"),
        "dual": KeyValueParameter("dual", bool, value=True),
        "tol": KeyValueParameter("tol", float, value=0.0001),
        "C": KeyValueParameter("C", float, value=1.0),
        "multi_class": KeyValueParameter("multi_class", str, value="ovr"),
        "fit_intercept": KeyValueParameter("fit_intercept", bool, value=True),
        "intercept_scaling": KeyValueParameter("intercept_scaling", int, value=1),
        "class_weight": KeyValueParameter("class_weight", float, value=None),
        "verbose": KeyValueParameter("verbose", int, value=0),
        "random_state": KeyValueParameter("random_state", str, value=None),
        "max_iter": KeyValueParameter("max_iter", int, value=1000),
    }

    def __init__(self, execute: list, **kwargs):
        super(SklearnLinearSVC, self).__init__(LinearSVC, execute, **kwargs)
