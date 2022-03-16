import pandas
import sklearn.base
from abc import abstractmethod

from rain.core.base import ComputationalNode, Tags, LibTag, TypeTag
from rain.core.exception import (
    EstimatorNotFoundException,
    InputNotFoundException,
)


class SklearnNode(ComputationalNode):
    _methods = {}

    def __init__(self, node_id):
        super(SklearnNode, self).__init__(node_id)
        self._estimator_or_function = None

    @abstractmethod
    def execute(self):
        raise NotImplementedError(
            "Method execute for class {} is not implemented yet.".format(
                self.__class__.__name__
            )
        )


class SklearnFunction(SklearnNode):
    def __init__(self, node_id: str):
        super(SklearnFunction, self).__init__(node_id)

    @abstractmethod
    def execute(self):
        raise NotImplementedError(
            "Method execute for class {} is not implemented yet.".format(
                self.__class__.__name__
            )
        )

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.SKLEARN, TypeTag.TRANSFORMER)


class SklearnEstimator(SklearnNode):
    _input_vars = {"fit_dataset": pandas.DataFrame}
    _methods = {"fit": False}
    _output_vars = {"fitted_model": sklearn.base.BaseEstimator}

    def __init__(self, node_id: str, execute: list):
        super(SklearnEstimator, self).__init__(node_id)

        for method in execute:
            if method not in self._methods.keys():
                raise Exception(
                    "Method {} not found for estimator {}".format(
                        method, self.__class__.__name__
                    )
                )

            self._methods[method] = True

    def fit(self):
        self.fitted_model = self._estimator_or_function.fit(self.fit_dataset)

    def execute(self):
        if self._estimator_or_function is None:
            raise EstimatorNotFoundException(
                "The estimator to use is not set for class {}".format(
                    self.__class__.__name__
                )
            )

        # se la fit deve essere eseguita, allora sarà sempre eseguita per prima
        if self._methods.get("fit") and self.fitted_model is None:
            self.fit()

        remaining_methods = [
            method
            for method, must_exec in self._methods.items()
            if must_exec and not method == "fit"
        ]

        for method_name in remaining_methods:
            method = eval("self.{}".format(method_name))
            method()

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.SKLEARN, TypeTag.ESTIMATOR)


class PredictorMixin:

    _input_vars = {"predict_dataset": pandas.DataFrame}

    _output_vars = {"predictions": pandas.DataFrame}

    _methods = {"predict": False}

    def predict(self):
        if self.fitted_model is not None and self.predict_dataset is not None:
            self.predictions = self.fitted_model.predict(self.predict_dataset)

            if (
                type(self.predictions) is not pandas.DataFrame
            ):  # some estimators returns a numpy ndarray
                self.predictions = pandas.DataFrame(self.predictions)


class ScorerMixin:

    _input_vars = {"score_dataset": pandas.DataFrame}

    _output_vars = {"score_value": float}

    _methods = {"score": False}

    def score(self):
        if self.score_dataset is None:
            raise InputNotFoundException(
                "The 'score_dataset' input is not set for node {}".format(
                    self.__class__.__name__
                )
            )

        if self.fitted_model is not None:
            if self._estimator_type == "classifier":
                if self.score_targets is None:
                    raise InputNotFoundException(
                        "The 'score_targets' input is not set for node {}".format(
                            self.__class__.__name__
                        )
                    )
                self.scores = self.fitted_model.score(
                    self.score_dataset, self.score_targets
                )
            else:
                self.scores = self.fitted_model.score(self.score_dataset)


class TransformerMixin:

    _input_vars = {"transform_dataset": pandas.DataFrame}

    _output_vars = {"transformed_dataset": pandas.DataFrame}

    _methods = {"transform": False}

    def transform(self):
        if self.transform_dataset is None:
            raise InputNotFoundException(
                "The 'transform_dataset' input is not set for node {}".format(
                    self.__class__.__name__
                )
            )

        if self.fitted_model is not None:
            self.transformed_dataset = self.fitted_model.transform(
                self.transform_dataset
            )

            if (
                type(self.transformed_dataset) is not pandas.DataFrame
            ):  # some estimators returns a numpy ndarray
                self.transformed_dataset = pandas.DataFrame(self.transformed_dataset)


class SklearnClassifier(SklearnEstimator, PredictorMixin, ScorerMixin):
    _estimator_type = "classifier"

    _input_vars = {"fit_targets": pandas.DataFrame, "score_targets": pandas.DataFrame}

    def __init__(self, node_id: str, execute: list):
        super(SklearnClassifier, self).__init__(node_id, execute)

    def fit(self):
        if self.fit_dataset is None:
            raise InputNotFoundException(
                "The 'fit_dataset' input is not set for node {}".format(
                    self.__class__.__name__
                )
            )
        elif self.fit_targets is None:
            raise InputNotFoundException(
                "The 'fit_targets' input is not set for node {}".format(
                    self.__class__.__name__
                )
            )

        self.fitted_model = self._estimator_or_function.fit(
            self.fit_dataset, self.fit_targets
        )

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.SKLEARN, TypeTag.CLASSIFIER)


class SklearnClusterer(SklearnEstimator, PredictorMixin, ScorerMixin, TransformerMixin):
    _estimator_type = "clusterer"

    def __init__(self, node_id: str, execute: list):
        super(SklearnClusterer, self).__init__(node_id, execute)

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.SKLEARN, TypeTag.CLUSTERER)
