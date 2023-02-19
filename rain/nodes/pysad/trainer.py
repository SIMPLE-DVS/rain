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

import numpy as np

from rain.core.parameter import Parameters, KeyValueParameter
from rain.nodes.pysad.node_structure import PySadTrainer

from pysad.models import IForestASD as IFASD, xStream as xS, HalfSpaceTrees as HST
from pysad.evaluation import AUROCMetric
from rain.loguru_logger import logger


class HalfSpaceTree(PySadTrainer):
    """Node that trains a model using the HalfSpaceTree algorithm.

    Input
    -----
    dataset : pd.DataFrame
        A Pandas DataFrame containing the features.
    labels : pd.Series
        A Pandas Series containing the labels.

    Output
    ------
    model : pickle
        The trained model in pickle format.
    auroc : float
        The AUROC metric of the trained model.

    Parameters
    ----------
    node_id : str
        Id of the node.
    window_size : int, default=100
        The size of the window.
    num_trees : int, default=25
        The number of trees.
    initial_window_x : np.ndarray, default=None
        The initial window to fit for initial calibration period. If not None, we simply apply fit to these instances.
    max_depth : int, default=15
        The maximum depth of the trees.
    """

    def __init__(self, node_id: str, data, window_size: int = 100, num_trees: int = 25, initial_window_x: np.ndarray = None, max_depth: int = 15):
        super(HalfSpaceTree, self).__init__(node_id)
        self.parameters = Parameters(
            window_size=KeyValueParameter("window_size", int, window_size),
            num_trees=KeyValueParameter("num_trees", int, num_trees),
            max_depth=KeyValueParameter("max_depth", int, max_depth),
            initial_window_X=KeyValueParameter("initial_window_X", np.ndarray, initial_window_x)

        )
        self.dataset = data
        self.model = HST(self.dataset.to_numpy().min(axis=0), self.dataset.to_numpy().max(axis=0), **self.parameters.get_dict())
        self.metric = AUROCMetric()

    def execute(self):
        self.scores = self.model.fit_score(self.dataset.to_numpy())
        if self.labels is not None:
            for label, score in zip(self.labels, self.scores):
                if np.isnan(score):
                    continue
                self.metric.update(label, score)
            self.auroc = self.metric.get()
            print(self.auroc)


class XStream(PySadTrainer):
    """Node that trains a model using the xStream algorithm.

    Input
    -----
    dataset : pd.DataFrame
        A Pandas DataFrame containing the features.
    labels : pd.Series
        A Pandas Series containing the labels.

    Output
    ------
    model : pickle
        The trained model in pickle format.
    auroc : float
        The AUROC metric of the trained model.

    Parameters
    ----------
    node_id : str
        Id of the node.
    window_size : int, default=25
        The size (and the sliding length) of the reference window.
    num_components : int, default=100
        The number of components for streamhash projection.
    n_chains : int, default=100
        The number of half-space chains.
    depth : int, default=25
        The maximum depth for the chains.
    """

    def __init__(self, node_id: str, window_size: int = 25, num_components: int = 100, n_chains: int = 100,
                 depth: int = 25):
        super(XStream, self).__init__(node_id)
        self.parameters = Parameters(
            window_size=KeyValueParameter("window_size", int, window_size),
            num_components=KeyValueParameter("num_components", int, num_components),
            n_chains=KeyValueParameter("n_chains", int, n_chains),
            depth=KeyValueParameter("depth", int, depth)

        )
        self.model = xS(**self.parameters.get_dict())
        self.metric = AUROCMetric()

    def execute(self):
        self.scores = self.model.fit_score(self.dataset.to_numpy())
        if self.labels is not None:
            for label, score in zip(self.labels, self.scores):
                if np.isnan(score):
                    continue
                self.metric.update(label, score)
            self.auroc = self.metric.get()
            print(self.auroc)


class IForestASD(PySadTrainer):
    """Node that trains a model using the IForestASD algorithm.

    Input
    -----
    dataset : pd.DataFrame
        A Pandas DataFrame containing the features.
    labels : pd.Series
        A Pandas Series containing the labels.

    Output
    ------
    model : pickle
        The trained model in pickle format.
    auroc : float
        The AUROC metric of the trained model.

    Parameters
    ----------
    node_id : str
        Id of the node.
    window_size : int, default= 2048
        The size of the reference window and its sliding.
    """

    def __init__(self, node_id: str, window_size: int = 2048):
        super(IForestASD, self).__init__(node_id)
        self.parameters = Parameters(
            window_size=KeyValueParameter("window_size", int, window_size)
        )
        self.model = IFASD(**self.parameters.get_dict())
        self.metric = AUROCMetric()

    def execute(self):
        self.scores = self.model.fit_score(self.dataset.to_numpy())
        if self.labels is not None:
            for label, score in zip(self.labels, self.scores):
                if np.isnan(score):
                    continue
                self.metric.update(label, score)
            self.auroc = self.metric.get()
            logger.info(f"Model trained - AUROC: {self.auroc}", node_name=self.node_id)
