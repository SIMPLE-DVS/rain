"""
 Copyright (C) 2023 Università degli Studi di Camerino and Sigma S.p.A.
 Authors: Alessandro Antinori, Rosario Capparuccia, Riccardo Coltrinari, Flavio Corradini, Marco Piangerelli, Barbara Re, Marco Scarpetta

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU Affero General Public License as
 published by the Free Software Foundation, either version 3 of the
 License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Affero General Public License for more details.

 You should have received a copy of the GNU Affero General Public License
 along with this program.  If not, see <https://www.gnu.org/licenses/>.
 """

import pandas
import sklearn.base
from abc import abstractmethod

from rain.core.base import ComputationalNode, Tags, LibTag, TypeTag
from rain.core.exception import (
    EstimatorNotFoundException,
    InputNotFoundException,
)


class SklearnNode(ComputationalNode):
    """Base class for all the nodes that use the sklearn library."""

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
    """Base class for all the nodes that use an sklearn function."""

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
    """Base class for all the nodes that use an sklearn Estimator.

    Input
    -----
    fitted_model : sklearn.base.BaseEstimator
        A previously fitted model.
    dataset : pandas.DataFrame
        The dataset that will be used to perform the different methods on.

    Output
    ------
    fitted_model : sklearn.base.BaseEstimator
        The model that results from the fit of the estimator.

    Parameters
    ----------
    node_id : str
        Id of the node.
    execute : [fit]
        List of strings to specify the methods to execute.
        The allowed strings are those from the _method attribute.
    """

    _input_vars = {"fitted_model": sklearn.base.BaseEstimator, "dataset": pandas.DataFrame}
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
        self.fitted_model = self._estimator_or_function.fit(self.dataset)

    def execute(self):
        if self._estimator_or_function is None:
            raise EstimatorNotFoundException(
                "The estimator to use is not set for class {}".format(
                    self.__class__.__name__
                )
            )

        # se la fit deve essere eseguita, allora sarà sempre eseguita per prima.
        # se la fit deve sovrascrivere il fitted_model, allora rimuovere la
        # seconda parte dell'if statement ("and self.fitted_model is None")
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
    """Mixin class to add a prediction functionality to an estimator."""

    _output_vars = {"predictions": pandas.DataFrame}

    _methods = {"predict": False}

    def predict(self):
        if self.fitted_model is not None and self.dataset is not None:
            self.predictions = self.fitted_model.predict(self.dataset)

            if (
                type(self.predictions) is not pandas.DataFrame
            ):  # some estimators returns a numpy ndarray
                self.predictions = pandas.DataFrame(self.predictions)


class ScorerMixin:
    """Mixin class to add a scoring functionality to an estimator."""

    _input_vars = {"score_targets": pandas.DataFrame}

    _output_vars = {"score_value": float}

    _methods = {"score": False}

    def score(self):
        if self.fitted_model is not None:
            if self._estimator_type == "classifier":
                if self.score_targets is None:
                    raise InputNotFoundException(
                        "The 'score_targets' input is not set for node {}".format(
                            self.__class__.__name__
                        )
                    )
                self.score_value = self.fitted_model.score(
                    self.dataset, self.score_targets
                )
            else:
                self.score_value = self.fitted_model.score(self.dataset)


class TransformerMixin:
    """Mixin class to add a transformer functionality to an estimator."""

    _output_vars = {"transformed_dataset": pandas.DataFrame}

    _methods = {"transform": False}

    def transform(self):
        if self.fitted_model is not None:
            self.transformed_dataset = self.fitted_model.transform(
                self.dataset
            )

            if (
                type(self.transformed_dataset) is not pandas.DataFrame
            ):  # some estimators returns a numpy ndarray
                self.transformed_dataset = pandas.DataFrame(self.dataset)


class SklearnClassifier(SklearnEstimator, PredictorMixin, ScorerMixin):
    """Base class for all the nodes that use an sklearn classifier.

    Input
    -----
    fitted_model : sklearn.base.BaseEstimator
        A previously fitted model.
    dataset : pandas.DataFrame
        The dataset to be used by the estimator.
    fit_targets : pandas.DataFrame
        The dataset that will be used as targets (labels) to perform the fit of the classifier.
    score_targets : pandas.DataFrame
        The dataset that will be used as targets (labels) to perform the scoring.

    Output
    ------
    fitted_model : sklearn.base.BaseEstimator
        The model that results from the fit of the estimator.
    predictions : pandas.DataFrame
        The predictions that result from the predict.
    score_value : float
        The score value that results from the scoring.

    Parameters
    ----------
    node_id : str
        Id of the node.
    execute : [fit, predict, score]
        List of strings to specify the methods to execute.
        The allowed strings are those from the _method attribute.
    """

    _estimator_type = "classifier"

    _input_vars = {"fit_targets": pandas.DataFrame}

    def __init__(self, node_id: str, execute: list):
        super(SklearnClassifier, self).__init__(node_id, execute)

    def fit(self):
        if self.dataset is None:
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
            self.dataset, self.fit_targets
        )

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.SKLEARN, TypeTag.CLASSIFIER)


class SklearnClusterer(SklearnEstimator, PredictorMixin, ScorerMixin, TransformerMixin):
    """Base class for all the nodes that use an sklearn clusterer.

    Input
    -----
    fitted_model : sklearn.base.BaseEstimator
        A previously fitted model.
    dataset : pandas.DataFrame
        The dataset to be used by the estimator.
    score_targets : pandas.DataFrame
        The dataset that will be used as targets (labels) to perform the scoring.

    Output
    ------
    fitted_model : sklearn.base.BaseEstimator
        The model that results from the fit of the estimator.
    predictions : pandas.DataFrame
        The predictions that result from the predict.
    score_value : float
        The score value that results from the scoring.
    transformed_dataset : pandas.DataFrame
        The dataset that results from the transform.

    Parameters
    ----------
    node_id : str
        Id of the node.
    execute : [fit, predict, score, transform]
        List of strings to specify the methods to execute.
        The allowed strings are those from the _method attribute.
    """

    _estimator_type = "clusterer"

    def __init__(self, node_id: str, execute: list):
        super(SklearnClusterer, self).__init__(node_id, execute)

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.SKLEARN, TypeTag.CLUSTERER)
