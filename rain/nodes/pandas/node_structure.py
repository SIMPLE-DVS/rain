from abc import abstractmethod

import pandas as pd

from rain.core.base import ComputationalNode, TypeTag, LibTag, Tags


class PandasTransformer(ComputationalNode):
    """Parent class for all the nodes that take a dataset as input, apply a transformation and expose the transformed dataset as output.

    Parameters
    ----------
    node_id : str
        Unique identifier of the node in the DataFlow.
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
    
    Parameters
    ----------
    node_id : str
        Unique identifier of the node in the DataFlow.
    """

    def __init__(self, node_id: str):
        super(PandasNode, self).__init__(node_id)

    @abstractmethod
    def execute(self):
        pass

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.PANDAS, TypeTag.TRANSFORMER)
