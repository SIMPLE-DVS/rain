from abc import abstractmethod

import pandas as pd

from simple_repo.base import ComputationalNode


class PandasNode(ComputationalNode):
    """
    Every PandasNode takes a pandas DataFrame as input,
    applies a tansformation and returns a pandas DataFrame as output.
    """

    _input_vars = {"dataset": pd.DataFrame}
    _parameters = {}
    _output_vars = {"dataset": pd.DataFrame}

    def __init__(self, **kwargs):
        super(PandasNode, self).__init__(**kwargs)

    @abstractmethod
    def execute(self):
        pass

    def __str__(self):
        return "{}".format(self._get_params_as_dict())
