from sklearn.decomposition import PCA

from simple_repo.parameter import Parameters, KeyValueParameter
from simple_repo.simple_sklearn.node_structure import (
    SklearnEstimator,
    TransformerMixin,
    ScorerMixin,
)


class SklearnPCA(SklearnEstimator, TransformerMixin, ScorerMixin):
    """
    svd_solver : {'auto', 'full', 'arpack', 'randomized'}, default='auto'
    """

    def __init__(
        self,
        node_id: str,
        execute: list,
        n_components=None,
        *,
        whiten=False,
        svd_solver="auto",
        tol=0.0,
        iterated_power="auto",
        random_state=None
    ):
        super(SklearnPCA, self).__init__(node_id, execute)
        self.parameters = Parameters(
            n_components=KeyValueParameter("n_components", int, n_components),
            whiten=KeyValueParameter("whiten", bool, whiten),
            svd_solver=KeyValueParameter("svd_solver", str, svd_solver),
            tol=KeyValueParameter("tol", float, tol),
            iterated_power=KeyValueParameter("iterated_power", str, iterated_power),
            random_state=KeyValueParameter("random_state", int, random_state),
        )
        self._estimator_or_function = PCA(**self.parameters.get_dict())
