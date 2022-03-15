from sklearn.datasets import load_iris

import pandas

from simple_repo.base import Tags, LibTag, TypeTag, InputNode


class IrisDatasetLoader(InputNode):
    """Loads the iris dataset as a pandas DataFrame."""

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
        return Tags(LibTag.PANDAS, TypeTag.INPUT)
