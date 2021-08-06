from node_structure import SklearnClassifier
from sklearn.svm import LinearSVC

import pandas


class SklearnLinearSVC(SklearnClassifier):
    _parameters = {
        "penalty": "l2",
        "loss": "squared_hinge",
        "dual": True,
        "tol": 0.0001,
        "C": 1.0,
        "multi_class": "ovr",
        "fit_intercept": True,
        "intercept_scaling": 1,
        "class_weight": None,
        "verbose": 0,
        "random_state": None,
        "max_iter": 1000,
    }

    def __init__(self, **kwargs):
        super(SklearnLinearSVC, self).__init__(LinearSVC, **kwargs)
