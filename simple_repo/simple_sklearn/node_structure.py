import csv
import inspect
import types
from abc import ABC, abstractmethod
import functools
from typing import List, Tuple, Any

import pandas
import sklearn.base
from sklearn.svm import LinearSVC
from sklearn.base import BaseEstimator


class ParameterNotFound(Exception):
    def __init__(self, msg: str):
        super(ParameterNotFound, self).__init__(msg)


class SklearnParameter:
    """
    An Sklearn Parameter contains information about parameters that can be used during the transformation.
    """

    def __init__(
        self,
        param_name: str,
        param_type: type,
        param_value: Any = None,
        is_mandatory: bool = False,
    ):
        self._param_name = param_name
        self._param_type = param_type
        self._param_value = param_value
        self._is_mandatory = is_mandatory

    @property
    def param_name(self):
        return self._param_name

    @property
    def param_type(self):
        return self._param_type

    @property
    def param_value(self):
        return self._param_value

    @param_value.setter
    def param_value(self, param_value):
        self._param_value = param_value

    def __str__(self):
        return "{{{}: {}}}".format(self._param_name, self._param_value)


class SklearnMethod:
    def __init__(self, method_name: str):
        self._method_name = method_name

    @property
    def method_name(self):
        return self._method_name


class SklearnEstimator:
    _input_vars = {"fit_dataset": pandas.DataFrame}
    _parameters = {}
    _methods = {"fit": SklearnMethod("fit")}
    _output_vars = {"fitted_model": sklearn.base.BaseEstimator}

    def _get_params_as_dict(self) -> dict:
        dct = {}
        for pval in self._parameters.values():
            dct[pval.param_name] = pval.param_value

        return dct

    def __init__(self, estimator_class: type, **kwargs):

        for k in self._input_vars.keys():
            setattr(self, k, None)

        # Set every output as an attribute if not already set
        for key in self._output_vars.keys():
            if key not in self._input_vars:
                setattr(self, key, None)

        # check the parameter passed and set their values
        for param_inst_name, param_inst_val in kwargs.items():
            try:
                # retrieve the parameter from its name
                par = self._parameters.get(param_inst_name)

                # if it is a parameter list add all the values inside, otherwise set the value of the parameter.
                if not isinstance(param_inst_val, par.param_type):
                    raise TypeError(
                        "Expected type '{}' for parameter '{}' in class '{}', received type '{}'.".format(
                            par.param_type,
                            param_inst_name,
                            self.__class__.__name__,
                            type(param_inst_val),
                        )
                    )
                else:
                    par.param_value = param_inst_val

            except AttributeError:
                raise ParameterNotFound(
                    "Class '{}' has no attribute '{}'".format(
                        self.__class__.__name__, param_inst_name
                    )
                )

        self._estimator = estimator_class(**kwargs)

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
        self._parameters["target_col"] = SklearnParameter(
            "target_col", str, is_mandatory=True
        )
        # self._input_vars["fit_target"] = pandas.DataFrame
        # self._input_vars["score_target"] = pandas.DataFrame
        super(SklearnClassifier, self).__init__(estimator_type, **kwargs)
