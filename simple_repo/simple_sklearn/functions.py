import pandas
from sklearn.model_selection import train_test_split

from simple_repo.parameter import Parameters, KeyValueParameter
from simple_repo.simple_sklearn.node_structure import SklearnFunction


class TrainTestDatasetSplit(SklearnFunction):
    """ """

    _input_vars = {"dataset": pandas.DataFrame}

    _output_vars = {
        "train_dataset": pandas.DataFrame,
        "test_dataset": pandas.DataFrame,
    }

    def __init__(
        self,
        node_id: str,
        test_size=None,
        train_size=None,
        random_state=None,
        shuffle: bool = True,
    ):
        super(TrainTestDatasetSplit, self).__init__(node_id)
        self.parameters = Parameters(
            test_size=KeyValueParameter("test_size", float, test_size),
            train_size=KeyValueParameter("train_size", float, train_size),
            random_state=KeyValueParameter("random_state", int, random_state),
            shuffle=KeyValueParameter("shuffle", bool, shuffle),
        )

    def execute(self):
        self.train_dataset, self.test_dataset = train_test_split(
            self.dataset, **self.parameters.get_dict()
        )


class TrainTestSampleTargetSplit(SklearnFunction):
    _input_vars = {
        "sample_dataset": pandas.DataFrame,
        "target_dataset": pandas.DataFrame,
    }

    _output_vars = {
        "sample_train_dataset": pandas.DataFrame,
        "sample_test_dataset": pandas.DataFrame,
        "target_train_dataset": pandas.DataFrame,
        "target_test_dataset": pandas.DataFrame,
    }

    def __init__(
        self,
        node_id: str,
        test_size=None,
        train_size=None,
        random_state=None,
        shuffle: bool = True,
    ):
        super(TrainTestSampleTargetSplit, self).__init__(node_id)
        self.parameters = Parameters(
            test_size=KeyValueParameter("test_size", float, test_size),
            train_size=KeyValueParameter("train_size", float, train_size),
            random_state=KeyValueParameter("random_state", int, random_state),
            shuffle=KeyValueParameter("shuffle", bool, shuffle),
        )

    def execute(self):
        (
            self.sample_train_dataset,
            self.sample_test_dataset,
            self.target_train_dataset,
            self.target_test_dataset,
        ) = train_test_split(
            self.sample_dataset, self.target_dataset, **self.parameters.get_dict()
        )
