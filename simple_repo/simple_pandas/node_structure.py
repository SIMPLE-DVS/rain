from abc import abstractmethod
import pandas as pd
from typing import List

from simple_repo.base import SimpleNode


class PandasNode(SimpleNode):
    """
    Every PandasNode takes a pandas DataFrame as input,
    applies a tansformation and returns a pandas DataFrame as output.
    """

    _input_vars = {
        "dataset": pd.DataFrame
    }
    _parameters = {}
    _output_vars = {
        "dataset": pd.DataFrame
    }

    def __init__(self, **kwargs):
        super(PandasNode, self).__init__(**kwargs)

    @abstractmethod
    def execute(self):
        pass

    def __str__(self):
        return "{}".format(self._get_params_as_dict())


class PandasPipeline:
    """
    PandasPipeline represents a sequence of transformation of a pandas dataframe.
    The nodes to use for the transformation are sent in a list of stages.
    The method transform is used to start the computation.
    """

    def __init__(self, stages: List[PandasNode]):
        self._stages = stages

    @property
    def stages(self):
        return self._stages

    def append_stage(self, stage: PandasNode):
        self._stages.append(stage)

    def execute(self):
        if len(self._stages) == 0:
            return None

        for i in range(0, len(self._stages)):
            self._stages[i].execute()

            if i + 1 > len(self._stages) - 1:
                break

            try:
                self._stages[i + 1].dataset = self._stages[i].dataset
                self._stages[i].dataset = None
            except Exception as e:
                print(e)

        return self._stages[len(self._stages) - 1].dataset
