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

from rain.core.parameter import KeyValueParameter, Parameters
from rain.nodes.sklearn.node_structure import SklearnClusterer
from sklearn.cluster import KMeans


class SimpleKMeans(SklearnClusterer):
    """A clusterer for the sklearn KMeans that uses the 'sklearn.cluster.KMeans'.

    Input
    -----
    fit_dataset : pandas.DataFrame
        The dataset that will be used to perform the fit of the clusterer.
    predict_dataset : pandas.DataFrame
        The dataset that will be used to perform the predict of the clusterer.
    score_dataset : pandas.DataFrame
        The dataset that will be used to perform the scoring.
    transform_dataset : pandas.DataFrame
        The dataset that will be used to perform the transform.

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
    labels : pandas.DataFrame
        Labels of each point.
        It corresponds to the 'labels_' attribute of the sklearn KMeans.

    Parameters
    ----------
    node_id : str
        Id of the node.
    execute : [fit, predict, score, transform]
        List of strings to specify the methods to execute.
        The allowed strings are those from the _method attribute.
    n_clusters : int
        The number of clusters to form as well as the number of centroids to generate.
    """

    _output_vars = {"labels": pandas.DataFrame}

    def __init__(self, node_id: str, execute: list, n_clusters: int = 8):
        super(SimpleKMeans, self).__init__(node_id, execute)
        self.parameters = Parameters(
            n_clusters=KeyValueParameter("n_clusters", int, n_clusters)
        )
        self._estimator_or_function = KMeans(**self.parameters.get_dict())

    def execute(self):
        super(SimpleKMeans, self).execute()
        self.labels = self.fitted_model.labels_
