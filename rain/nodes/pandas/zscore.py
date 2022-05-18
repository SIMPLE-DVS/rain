import pickle
from typing import List
import numpy as np
import pandas as pd
from rain.core.parameter import Parameters, KeyValueParameter
from rain.nodes.pandas.node_structure import PandasNode


class ZScoreTrainer(PandasNode):
    """Node that returns the model trained with the ZScore algorithm by analyzing the columns of the dataset.

    Input
    -----
    dataset : pandas.DataFrame
        The pandas DataFrame.

    Output
    ------
    model : pickle
        The ZScore model in pickle format.

    Parameters
    ----------
    columns : List[str]
        Column names to apply ZScore to. Empty to use all columns.
    """

    _input_vars = {"dataset": pd.DataFrame}

    _output_vars = {"model": "pickle"}

    def __init__(self, node_id: str, columns: List[str] = []):
        super(ZScoreTrainer, self).__init__(node_id)

        self.parameters = Parameters(
            columns=KeyValueParameter("columns", List[str], columns),
        )

    def execute(self):
        if not self.parameters.columns.value:
            self.parameters.columns.value = self.dataset.columns
        mean = {}
        dev_std = {}
        for column in self.parameters.columns.value:
            content = self.dataset[column]
            mean[column] = np.mean(content)
            dev_std[column] = np.std(content)
        self.model = pickle.dumps({"mean": mean, "dev_std": dev_std})


class ZScorePredictor(PandasNode):
    """Node that returns the predictions performed with a ZScore model on the columns of a dataset.

    Input
    -----
    dataset : pandas.DataFrame
        The pandas DataFrame.

    model : pickle
        The ZScore model in pickle format.

    Output
    ------
    predictions : pandas.DataFrame
        The DataFrame containing the predictions.

    Parameters
    ----------
    columns : List[str]
        Column names to apply ZScore to. Empty to use all columns.

    threshold : float, default=1.3,
        The threshold of the ZScore to distinguish anomalies.
    """

    _input_vars = {"dataset": pd.DataFrame, "model": "pickle"}

    _output_vars = {"predictions": pd.DataFrame}

    def __init__(self, node_id: str, columns: List[str] = [], threshold: float = 1.3):
        super(ZScorePredictor, self).__init__(node_id)

        self.parameters = Parameters(
            columns=KeyValueParameter("columns", List[str], columns),
            threshold=KeyValueParameter("threshold", float, threshold),
        )
        self.predictions = {}

    def execute(self):
        if not self.parameters.columns.value:
            self.parameters.columns.value = self.dataset.columns
        model = pickle.loads(self.model)
        for column in self.parameters.columns.value:
            self.predictions[column] = []
            content = self.dataset[column]
            for i in content:
                z = (i - model.get("mean")[column]) / model.get("dev_std")[column]
                if abs(z) > self.parameters.threshold.value:
                    self.predictions[column].append(1)
                else:
                    self.predictions[column].append(0)

        self.predictions = pd.DataFrame.from_dict(self.predictions)
