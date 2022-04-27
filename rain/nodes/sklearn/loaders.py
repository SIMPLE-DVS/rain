from sklearn.datasets import load_iris

import pandas

from rain.core.base import Tags, LibTag, TypeTag, InputNode


class IrisDatasetLoader(InputNode):
    """Loads the iris dataset as a pandas DataFrame using the 'sklearn.datasets.load_iris'.

    Output
    ------
    dataset : pandas.DataFrame
        The iris dataset.
    target : pandas.DataFrame
        If separate_target is enabled then it will contain the target labels for the iris dataset.

    Parameters
    ----------
    node_id : str
        Id of the node.
    separate_target : bool, default='False'
        Whether to get the target labels in the separated output 'target'.
    """

    _output_vars = {"dataset": pandas.DataFrame, "target": pandas.DataFrame}

    def __init__(self, node_id: str, separate_target: bool = False):
        self._separate_target = separate_target
        super(IrisDatasetLoader, self).__init__(node_id)

    def execute(self):
        if self._separate_target:
            self.dataset, self.target = load_iris(return_X_y=True, as_frame=True)
        else:
            self.dataset = load_iris(as_frame=True).data

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.SKLEARN, TypeTag.INPUT)
