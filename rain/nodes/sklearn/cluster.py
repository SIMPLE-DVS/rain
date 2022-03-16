import pandas

from rain.core.parameter import KeyValueParameter, Parameters
from rain.nodes.sklearn.node_structure import SklearnClusterer
from sklearn.cluster import KMeans


class SimpleKMeans(SklearnClusterer):
    """A clusterer for the sklearn KMeans.

    Parameters
    ----------
    execute : list[str]
        Methods to execute with this clusterer, they can be: fit, predict, transform, score.
    n_clusters : int
        The number of clusters to form as well as the number of centroids to generate.
    """

    _output_vars = {"labels": pandas.DataFrame}

    def __init__(self, node_id: str, execute: list, n_clusters: int = 8):
        super(SimpleKMeans, self).__init__(node_id, execute)
        self.parameters = Parameters(
            n_clusters=KeyValueParameter("n_clusters", int, n_clusters)
        )
        self._estimator_or_function = KMeans(**self.parameters.get_dict())

    def execute(self):
        super(SimpleKMeans, self).execute()
        self.labels = self.fitted_model.labels_
