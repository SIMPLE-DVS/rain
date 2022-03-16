from abc import abstractmethod

import pandas as pd

from simple_repo.core.base import ComputationalNode, TypeTag, LibTag, Tags


class PandasTransformer(ComputationalNode):
    """
    Every PandasNode takes a pandas DataFrame as input,
    applies a tansformation and returns a pandas DataFrame as output.
    """

    _input_vars = {"dataset": pd.DataFrame}
    _output_vars = {"dataset": pd.DataFrame}

    def __init__(self, node_id: str):
        super(PandasTransformer, self).__init__(node_id)

    @abstractmethod
    def execute(self):
        pass

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.PANDAS, TypeTag.TRANSFORMER)


class PandasNode(ComputationalNode):
    """
    Node that perform some transformation using the Pandas library without input/output constraints.
    """

    def __init__(self, node_id: str):
        super(PandasNode, self).__init__(node_id)

    @abstractmethod
    def execute(self):
        pass

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.PANDAS, TypeTag.TRANSFORMER)
