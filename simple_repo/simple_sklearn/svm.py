from sklearn.svm import LinearSVC

from simple_repo.parameter import KeyValueParameter, Parameters
from simple_repo.simple_sklearn.node_structure import SklearnClassifier


class SklearnLinearSVC(SklearnClassifier):
    def __init__(
        self,
        node_id: str,
        execute: list,
        penalty: str = "l2",
        loss: str = "squared_hinge",
        dual: bool = True,
        tol: float = 0.0001,
        C: float = 1.0,
        multi_class: str = "ovr",
        fit_intercept: bool = True,
        intercept_scaling: int = 1,
        class_weight: float = None,
        verbose: int = 0,
        random_state: str = None,
        max_iter: int = 1000,
    ):
        super(SklearnLinearSVC, self).__init__(node_id, execute)
        self.parameters = Parameters(
            penalty=KeyValueParameter("penalty", str, penalty),
            loss=KeyValueParameter("loss", str, loss),
            dual=KeyValueParameter("dual", bool, dual),
            tol=KeyValueParameter("tol", float, tol),
            C=KeyValueParameter("C", float, C),
            multi_class=KeyValueParameter("multi_class", str, multi_class),
            fit_intercept=KeyValueParameter("fit_intercept", bool, fit_intercept),
            intercept_scaling=KeyValueParameter(
                "intercept_scaling", int, intercept_scaling
            ),
            class_weight=KeyValueParameter("class_weight", float, class_weight),
            verbose=KeyValueParameter("verbose", int, verbose),
            random_state=KeyValueParameter("random_state", str, random_state),
            max_iter=KeyValueParameter("max_iter", int, max_iter),
        )

        self._estimator_or_function = LinearSVC(**self.parameters.get_dict())
