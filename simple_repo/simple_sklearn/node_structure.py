from typing import List, Tuple

import pandas
import sklearn.base

from simple_repo.base import SimpleNode
from simple_repo.parameter import KeyValueParameter


class SklearnMethod:
    def __init__(self, method_name: str):
        self._method_name = method_name

    @property
    def method_name(self):
        return self._method_name


class SklearnNode(SimpleNode):
    _methods = {}

    def __init__(self, **kwargs):
        super(SklearnNode, self).__init__(**kwargs)

    def execute(self):
        pass


class SklearnFunction(SklearnNode):
    def __init__(self, **kwargs):
        super(SklearnFunction, self).__init__(**kwargs)


class SklearnEstimator(SklearnNode):
    _input_vars = {
        "fit_dataset": pandas.DataFrame
    }
    _parameters = {}
    _methods = {
        "fit": SklearnMethod("fit")
    }
    _output_vars = {
        "fitted_model": sklearn.base.BaseEstimator
    }

    def __init__(self, estimator_class: type, **kwargs):
        super(SklearnEstimator, self).__init__(**kwargs)
        self._estimator = estimator_class(**self._get_params_as_dict())

    def fit(self):
        if self._estimator_type == "classifier" and self.fit_target is not None:
            self.fitted_model = self._estimator.fit(self.fit_dataset, self.fit_target)
        else:
            self.fitted_model = self._estimator.fit(self.fit_dataset)


class PredictorMixin:
    def __init__(self):
        self._input_vars["predict_dataset"] = pandas.DataFrame
        self._output_vars["predictions"] = pandas.DataFrame
        self._methods["predict"] = SklearnMethod("predict")

    def predict(self):
        if self.fitted_model is not None and self.predict_dataset is not None:
            self.predictions = self.fitted_model.predict(self.predict_dataset)


class ScorerMixin:
    def __init__(self):
        self._input_vars["score_dataset"] = pandas.DataFrame
        self._output_vars["scores"] = List[Tuple]
        self._methods["score"] = SklearnMethod("score")

    def score(self):
        if self.fitted_model is not None and self.score_dataset is not None:
            if self.score_target is not None:
                self.scores = self.fitted_model.score(
                    self.score_dataset, self.score_target
                )
            else:
                self.scores = self.fitted_model.score(self.score_dataset)


class SklearnClassifier(SklearnEstimator, PredictorMixin, ScorerMixin):
    _estimator_type = "classifier"

    def __init__(self, estimator_type: type, **kwargs):
        PredictorMixin.__init__(self)
        ScorerMixin.__init__(self)
        # self._parameters["target_col"] = KeyValueParameter("target_col", str, is_mandatory=True)
        # self._input_vars["fit_target"] = pandas.DataFrame
        # self._input_vars["score_target"] = pandas.DataFrame
        super(SklearnClassifier, self).__init__(estimator_type, **kwargs)
