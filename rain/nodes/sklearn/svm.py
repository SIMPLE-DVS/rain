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

from sklearn.svm import LinearSVC

from rain.core.parameter import KeyValueParameter, Parameters
from rain.nodes.sklearn.node_structure import SklearnClassifier


class SklearnLinearSVC(SklearnClassifier):
    """Node that uses the 'sklearn.svm.LinearSVC' classifier.

    Input
    -----
    fit_dataset : pandas.DataFrame
        The dataset that will be used to perform the fit of the classifier.
    fit_targets : pandas.DataFrame
        The dataset that will be used as targets (labels) to perform the fit of the classifier.
    predict_dataset : pandas.DataFrame
        The dataset that will be used to perform the predict of the classifier.
    score_dataset : pandas.DataFrame
        The dataset that will be used to perform the scoring.
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
    execute : {'fit', 'predict', 'score'}
        List of strings to specify the methods to execute.
        The allowed strings are those from the _method attribute.
    penalty : str, default='l2'
        Penalty.
    loss : str, default='squared_hinge',
        Loss.
    dual : bool, default='True',
        Dual.
    tol : float, default='0.0001',
        Tol.
    C : float, default='1.0',
        C.
    multi_class : str, default='ovr',
        Multi_class.
    fit_intercept : bool, default='True',
        Fit_intercept.
    intercept_scaling : int, default='1',
        Intercept_scaling.
    class_weight : float, default='None',
        Class_weight.
    verbose : int, default='0',
        Verbose.
    random_state : str, default='None',
        Random_state.
    max_iter : int, default='1000',
        Max_iter.
    """

    def __init__(
        self,
        node_id: str,
        execute: list,
        penalty: str = "l2",
        loss: str = "squared_hinge",
        dual: bool = True,
        tol: float = 0.0001,
        C: float = 1.0,
        multi_class: str = "ovr",
        fit_intercept: bool = True,
        intercept_scaling: int = 1,
        class_weight: float = None,
        verbose: int = 0,
        random_state: str = None,
        max_iter: int = 1000,
    ):
        super(SklearnLinearSVC, self).__init__(node_id, execute)
        self.parameters = Parameters(
            penalty=KeyValueParameter("penalty", str, penalty),
            loss=KeyValueParameter("loss", str, loss),
            dual=KeyValueParameter("dual", bool, dual),
            tol=KeyValueParameter("tol", float, tol),
            C=KeyValueParameter("C", float, C),
            multi_class=KeyValueParameter("multi_class", str, multi_class),
            fit_intercept=KeyValueParameter("fit_intercept", bool, fit_intercept),
            intercept_scaling=KeyValueParameter(
                "intercept_scaling", int, intercept_scaling
            ),
            class_weight=KeyValueParameter("class_weight", float, class_weight),
            verbose=KeyValueParameter("verbose", int, verbose),
            random_state=KeyValueParameter("random_state", str, random_state),
            max_iter=KeyValueParameter("max_iter", int, max_iter),
        )

        self._estimator_or_function = LinearSVC(**self.parameters.get_dict())
