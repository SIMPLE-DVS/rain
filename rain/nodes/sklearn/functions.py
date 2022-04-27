import pandas
from sklearn.metrics import davies_bouldin_score
from sklearn.model_selection import train_test_split

from rain.core.base import TypeTag, LibTag, Tags
from rain.core.parameter import Parameters, KeyValueParameter
from rain.nodes.sklearn.node_structure import SklearnFunction


class TrainTestDatasetSplit(SklearnFunction):
    """Node that uses the 'sklearn.model_selection.train_test_split' to split a dataset in two parts.

    Input
    -----
    dataset : pandas.DataFrame
        The dataset to split.

    Output
    ------
    train_dataset : pandas.DataFrame
        The training dataset.
    test_dataset : pandas.DataFrame
        The test dataset.

    Parameters
    ----------
    node_id : str
        Id of the node.
    test_size: float = None
        The size as percentage of the test dataset (e.g. 0.3 is 30%).
    train_size: float = None
        The size as percentage of the train dataset (e.g. 0.7 is 70%)
    random_state: int = None
        Seed for the random generation.
    shuffle : bool = True
        Whether to shuffle the dataset before the splitting.
    """

    _input_vars = {"dataset": pandas.DataFrame}

    _output_vars = {
        "train_dataset": pandas.DataFrame,
        "test_dataset": pandas.DataFrame,
    }

    def __init__(
        self,
        node_id: str,
        test_size: float = None,
        train_size: float = None,
        random_state: int = None,
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
    """Node that uses the 'sklearn.model_selection.train_test_split' to split two datasets in four parts.
    It is useful for classification where you have to split equally the sample and the target datasets.

    Input
    -----
    sample_dataset : pandas.DataFrame
        The dataset containing the samples.
    target_dataset: pandas.DataFrame
        The dataset containing the target labels.

    Output
    ------
    sample_train_dataset : pandas.DataFrame
        The training dataset containing the samples.
    sample_test_dataset : pandas.DataFrame
        The test dataset containing the samples.
    target_train_dataset : pandas.DataFrame
        The training dataset containing the target labels.
    target_test_dataset : pandas.DataFrame
        The test dataset containing the target labels.

    Parameters
    ----------
    node_id : str
        Id of the node.
    test_size: float = None
        The size as percentage of the test dataset (e.g. 0.3 is 30%).
    train_size: float = None
        The size as percentage of the train dataset (e.g. 0.7 is 70%)
    random_state: int = None
        Seed for the random generation.
    shuffle : bool = True
        Whether to shuffle the dataset before the splitting.
    """

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
    Computes the Davies-Bouldin score using the 'sklearn.metrics.davies_bouldin_score'.
    The score is defined as the average similarity measure of each cluster with its most similar cluster, where similarity is the ratio of within-cluster distances to between-cluster distances. Thus, clusters which are farther apart and less dispersed will result in a better score.
    The minimum score is zero, with lower values indicating better clustering.

    Input
    -----
    samples_dataset : pandas.DataFrame
        The dataset containing the samples.
    labels : pandas.DataFrame
        The dataset containing the target labels.

    Output
    ------
    score : float
        The davies boulding score value.

    Parameters
    ----------
    node_id : str
        Id of the node.
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
