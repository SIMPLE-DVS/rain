import pandas
from sklearn.metrics import davies_bouldin_score
from sklearn.model_selection import train_test_split

from simple_repo.core.base import TypeTag, LibTag, Tags
from simple_repo.core.parameter import Parameters, KeyValueParameter
from simple_repo.nodes.sklearn.node_structure import SklearnFunction


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


class DaviesBouldinScore(SklearnFunction):
    """
    Computes the Davies-Bouldin score.
    The score is defined as the average similarity measure of each cluster with its most similar cluster, where similarity is the ratio of within-cluster distances to between-cluster distances. Thus, clusters which are farther apart and less dispersed will result in a better score.
    The minimum score is zero, with lower values indicating better clustering.
    """

    _input_vars = {"samples_dataset": pandas.DataFrame, "labels": pandas.DataFrame}

    _output_vars = {"score": float}

    def __init__(self, node_id: str):
        super(DaviesBouldinScore, self).__init__(node_id)

    def execute(self):
        self.score = davies_bouldin_score(self.samples_dataset, self.labels)

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.SKLEARN, TypeTag.METRICS)
