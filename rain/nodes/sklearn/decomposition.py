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

from sklearn.decomposition import PCA

from rain.core.parameter import Parameters, KeyValueParameter
from rain.nodes.sklearn.node_structure import (
    SklearnEstimator,
    TransformerMixin,
    ScorerMixin,
)


class SklearnPCA(SklearnEstimator, TransformerMixin, ScorerMixin):
    """
    Node representation of a sklearn PCA estimator that uses the 'sklearn.decomposition.PCA'.

    Input
    -----
    fit_dataset : pandas.DataFrame
        The dataset that will be used to perform the fit of the clusterer.
    score_dataset : pandas.DataFrame
        The dataset that will be used to perform the scoring.
    transform_dataset : pandas.DataFrame
        The dataset that will be used to perform the transform.

    Output
    ------
    fitted_model : sklearn.base.BaseEstimator
        The model that results from the fit of the estimator.
    score_value : float
        The score value that results from the scoring.
    transformed_dataset : pandas.DataFrame
        The dataset that results from the transform.

    Parameters
    ----------
    execute : {'fit', 'score', 'transform'}
        List of strings to specify the methods to execute.
        The allowed strings are those from the _method attribute.
    n_components : int
        Number of components to keep.
    whiten : bool
        When True (False by default) the components_ vectors are multiplied by the square root of n_samples and then divided by the singular values to ensure uncorrelated outputs with unit component-wise variances.
    svd_solver : {'auto', 'full', 'arpack', 'randomized'}, default='auto'
        Svd solver.
    tol : float
        Tolerance for singular values computed by svd_solver == 'arpack'.
        Must be positive.
    iterated_power : int
        Number of iterations for the power method computed by svd_solver == 'randomized'.
        Must be positive.
    random_state : int
        Used when the 'arpack' or 'randomized' solvers are used. Pass an int for reproducible results across multiple function calls.
    """

    def __init__(
        self,
        node_id: str,
        execute: list,
        n_components=None,
        *,
        whiten=False,
        svd_solver="auto",
        tol=0.0,
        iterated_power="auto",
        random_state=None
    ):
        super(SklearnPCA, self).__init__(node_id, execute)
        self.parameters = Parameters(
            n_components=KeyValueParameter("n_components", int, n_components),
            whiten=KeyValueParameter("whiten", bool, whiten),
            svd_solver=KeyValueParameter("svd_solver", str, svd_solver),
            tol=KeyValueParameter("tol", float, tol),
            iterated_power=KeyValueParameter("iterated_power", str, iterated_power),
            random_state=KeyValueParameter("random_state", int, random_state),
        )
        self._estimator_or_function = PCA(**self.parameters.get_dict())
