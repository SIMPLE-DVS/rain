from simple_repo.parameter import KeyValueParameter
from simple_repo.simple_sklearn.node_structure import SklearnClusterer
from sklearn.cluster import KMeans


class SimpleKMeans(SklearnClusterer):
    _parameters = {"n_clusters": KeyValueParameter("n_clusters", int, value=8)}

    def __init__(self, execute: list, **kwargs):
        super(SimpleKMeans, self).__init__(KMeans, execute, **kwargs)
