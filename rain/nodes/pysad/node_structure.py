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

from abc import abstractmethod

from rain import ComputationalNode, Tags, LibTag, TypeTag
import pandas as pd


class PySadNode(ComputationalNode):
    """
    Node that perform some operations using the PySad library without input/output constraints.

    Parameters
    ----------
    node_id : str
        Unique identifier of the node in the DataFlow.
    """

    def __init__(self, node_id: str):
        super(PySadNode, self).__init__(node_id)

    @abstractmethod
    def execute(self):
        pass


class PySadTransformer(PySadNode):
    """Class representing a PySad Transformer, it manipulates a given dataset and returns a modified version of it.

    Input
    -----
    dataset : pd.DataFrame
        A Pandas DataFrame.

    Output
    ------
    dataset : pd.DataFrame
        A Pandas DataFrame.

    Parameters
    ----------
    node_id : str
        Id of the node.
    """

    _input_vars = {"dataset": pd.DataFrame}
    _output_vars = {"dataset": pd.DataFrame}

    def __init__(self, node_id: str):
        super(PySadTransformer, self).__init__(node_id)

    @abstractmethod
    def execute(self):
        pass

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.PYSAD, TypeTag.TRANSFORMER)


class PySadTrainer(PySadNode):
    """Class representing a PySad Trainer, it trains a model given a Dataset.

    Input
    -----
    dataset : pd.DataFrame
        A Pandas DataFrame.
    labels : pd.Series
        A Pandas Series containing the labels.

    Output
    ------
    model : pickle
        The trained model in pickle format.
    auroc : float
        The AUROC associated to the trained model.

    Parameters
    ----------
    node_id : str
        Id of the node.
    """

    _input_vars = {"dataset": pd.DataFrame, "labels": pd.Series}
    _output_vars = {"model": "pickle", "auroc": float}

    def __init__(self, node_id: str):
        super(PySadTrainer, self).__init__(node_id)

    @abstractmethod
    def execute(self):
        pass  # pragma: no cover

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.PYSAD, TypeTag.TRAINER)


class PySadPredictor(PySadNode):
    """Class representing a PySad predictor, use the given model and dataset to obtain the predictions.

    Input
    -----
    dataset : pd.DataFrame
        A Pandas DataFrame.
    model : pickle
        A model in pickle format.

    Output
    ------
    predictions : pd.DataFrame
        The DataFrame containing the predictions.

    Parameters
    ----------
    node_id : str
        Id of the node.
    """

    _input_vars = {"dataset": pd.DataFrame, "model": "pickle"}
    _output_vars = {"predictions": pd.DataFrame}

    def __init__(self, node_id: str):
        super(PySadPredictor, self).__init__(node_id)

    @abstractmethod
    def execute(self):
        self.predictions = self.model.fit_score(self.dataset.to_numpy())

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.PYSAD, TypeTag.PREDICTOR)
