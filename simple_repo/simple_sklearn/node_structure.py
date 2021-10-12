import pandas
import sklearn.base

from simple_repo.base import ComputationalNode


class SklearnMethod:
    def __init__(self, method_name: str, execute: bool = False):
        self._method_name = method_name

    @property
    def method_name(self):
        return self._method_name


class SklearnNode(ComputationalNode):
    _methods = {}

    def __init__(self, node_id):
        super(SklearnNode, self).__init__(node_id)

    def execute(self):
        pass


class SklearnFunction(SklearnNode):
    def __init__(self, node_id: str, **kwargs):
        super(SklearnFunction, self).__init__(node_id)


class SklearnEstimator(SklearnNode):
    _input_vars = {"fit_dataset": pandas.DataFrame}
    _methods = {"fit": False}
    _output_vars = {"fitted_model": sklearn.base.BaseEstimator}

    def __init__(self, node_id: str, estimator_class: type, execute: list):
        super(SklearnEstimator, self).__init__(node_id)
        self._estimator = estimator_class(**self._get_params_as_dict())

        for method in execute:
            if method not in self._methods.keys():
                raise Exception(
                    "Method {} not found for estimator {}".format(
                        method, self.__class__.__name__
                    )
                )

            self._methods[method] = True

    def fit(self):
        if self._estimator_type == "classifier" and self.fit_target is not None:
            self.fitted_model = self._estimator.fit(self.fit_dataset, self.fit_target)
        else:
            self.fitted_model = self._estimator.fit(self.fit_dataset)

    def execute(self):
        # se la fit deve essere eseguita, allora sarà sempre eseguita per prima
        if self._methods.get("fit"):
            self.fit()

        remaining_methods = [
            method
            for method, must_exec in self._methods.items()
            if must_exec and not method == "fit"
        ]

        for method_name in remaining_methods:
            method = eval("self.{}".format(method_name))
            method()


class PredictorMixin:

    _input_vars = {"predict_dataset": pandas.DataFrame}

    _output_vars = {"predictions": pandas.DataFrame}

    _methods = {"predict": False}

    def predict(self):
        if self.fitted_model is not None and self.predict_dataset is not None:
            self.predictions = self.fitted_model.predict(self.predict_dataset)


class ScorerMixin:

    _input_vars = {"score_dataset": pandas.DataFrame}

    _output_vars = {"scores": list}

    _methods = {"score": False}

    def score(self):
        if self.fitted_model is not None and self.score_dataset is not None:
            if self.score_target is not None:
                self.scores = self.fitted_model.score(
                    self.score_dataset, self.score_target
                )
            else:
                self.scores = self.fitted_model.score(self.score_dataset)


class TransformerMixin:

    _input_vars = {"transform_dataset": pandas.DataFrame}

    _output_vars = {"transformed_dataset": list}

    _methods = {"transform": False}

    def transform(self):
        if self.fitted_model is not None and self.transform_dataset is not None:
            self.transformed_dataset = self.fitted_model.score(self.transform_dataset)


class SklearnClassifier(SklearnEstimator, PredictorMixin, ScorerMixin):
    _estimator_type = "classifier"

    def __init__(self, node_id: str, estimator_type: type, execute: list):
        super(SklearnClassifier, self).__init__(node_id, estimator_type, execute)


class SklearnClusterer(SklearnEstimator, PredictorMixin, ScorerMixin, TransformerMixin):
    _estimator_type = "clusterer"

    def __init__(self, node_id: str, estimator_type: type, execute: list):
        super(SklearnClusterer, self).__init__(node_id, estimator_type, execute)
