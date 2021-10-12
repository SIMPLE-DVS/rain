from abc import abstractmethod

import pandas as pd

from simple_repo.base import ComputationalNode


class PandasNode(ComputationalNode):
    """
    Every PandasNode takes a pandas DataFrame as input,
    applies a tansformation and returns a pandas DataFrame as output.
    """

    _input_vars = {"dataset": pd.DataFrame}
    _output_vars = {"dataset": pd.DataFrame}

    def __init__(self, node_id: str):
        super(PandasNode, self).__init__(node_id)

    @abstractmethod
    def execute(self):
        pass
