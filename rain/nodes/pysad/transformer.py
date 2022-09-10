import pandas as pd

from rain.core.parameter import Parameters, KeyValueParameter
from rain.nodes.pysad.node_structure import PySadTransformer, PySadNode
from pysad.transform.preprocessing import InstanceUnitNormScaler as IUNScaler
from pysad.transform.probability_calibration import ConformalProbabilityCalibrator as Cpc, GaussianTailProbabilityCalibrator as Gtpc


class InstanceUnitNormScaler(PySadTransformer):
    """A scaler that makes the instance feature vector's norm equal to 1, i.e., the unit vector.

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
    pow : float, default=2
        The power, for which the norm is calculated. pow=2 is equivalent to the euclidean distance.
    """

    def __init__(self, node_id: str, pow: int = 2):
        super(InstanceUnitNormScaler, self).__init__(node_id)

        self.parameters = Parameters(
            pow=KeyValueParameter("pow", str, pow)
        )
        self.transformer = IUNScaler(**self.parameters.get_dict())

    def execute(self):
        columns = self.dataset.columns
        self.dataset = self.transformer.fit_transform(self.dataset.to_numpy())
        self.dataset = pd.DataFrame(self.dataset, columns=columns)


class ConformalProbabilityCalibrator(PySadNode):
    """This class provides an interface to convert the scores into probabilities through conformal prediction.

    Input
    -----
    scores : pd.DataFrame
        A Pandas DataFrame containing the scores.

    Output
    ------
    scores : pd.DataFrame
        A Pandas DataFrame containing the scores.

    Parameters
    ----------
    node_id : str
        Id of the node.
    windowed : bool, default=True
        Whether the probability calibrator is windowed so that forget scores that are older than `window_size`.
    window_size : int, default=300
        The size of window for running average and std.
    """

    _input_vars = {"scores": pd.DataFrame}
    _output_vars = {"scores": pd.DataFrame}

    def __init__(self, node_id: str, windowed: bool = True, window_size: int = 300):
        super(ConformalProbabilityCalibrator, self).__init__(node_id)

        self.parameters = Parameters(
            windowed=KeyValueParameter("windowed", bool, windowed),
            window_size=KeyValueParameter("window_size", int, window_size)
        )
        self.calibrator = Cpc(**self.parameters.get_dict())

    def execute(self):
        self.scores = self.calibrator.fit_transform(self.scores)


class GaussianTailProbabilityCalibrator(PySadNode):
    """This class provides an interface to convert the scores into probabilities via Q-function, i.e., the tail
     function of Gaussian distribution.

    Input
    -----
    scores : pd.DataFrame
        A Pandas DataFrame containing the scores.

    Output
    ------
    scores : pd.DataFrame
        A Pandas DataFrame containing the scores.

    Parameters
    ----------
    node_id : str
        Id of the node.
    running_statistics : bool, default=True
        Whether to calculate the mean and variance through running window.
    window_size : int, default=300
        The size of window for running average and std. Ignored if `running_statistics` parameter is False.
    """

    _input_vars = {"scores": pd.DataFrame}
    _output_vars = {"scores": pd.DataFrame}

    def __init__(self, node_id: str, running_statistics: bool = True, window_size: int = 300):
        super(GaussianTailProbabilityCalibrator, self).__init__(node_id)

        self.parameters = Parameters(
            running_statistics=KeyValueParameter("windowed", bool, running_statistics),
            window_size=KeyValueParameter("window_size", int, window_size)
        )
        self.calibrator = Cpc(**self.parameters.get_dict())

    def execute(self):
        self.scores = self.calibrator.fit_transform(self.scores)
