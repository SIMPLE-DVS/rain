import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import simple_repo.simple_sklearn as ss
from simple_repo.simple_sklearn.node_structure import SklearnNode

sklearn_nodes = [ss.SklearnLinearSVC, ss.SimpleKMeans]


@pytest.mark.parametrize("class_or_obj", sklearn_nodes)
def test_sklearn_methods(class_or_obj):
    """Checks whether the spark node has no method attributes."""
    assert issubclass(class_or_obj, SklearnNode) and hasattr(class_or_obj, "_methods")


@pytest.mark.parametrize("class_or_obj", sklearn_nodes)
def test_sklearn_execute(class_or_obj):
    """Checks whether the spark node has no method attributes."""
    with pytest.raises(Exception):
        class_or_obj("", execute=["non_existing_method"])


class TestKMeans:
    def test_execution(self):
        iris = load_iris(as_frame=True).data
        iristrain, iristest = train_test_split(iris, test_size=0.15, shuffle=False)
        node = ss.SimpleKMeans("km", ["fit", "score"])
        node.set_input_value("fit_dataset", iristrain)
        node.set_input_value("score_dataset", iristest)

        node.execute()

        print(node.scores)


class TestSklearnLinearSVC:
    def test_execution_passed_target(self):
        X, y = load_iris(return_X_y=True, as_frame=True)
        xtrain, xtest, ytrain, ytest = train_test_split(
            X, y, test_size=0.15, shuffle=False
        )

        node = ss.SklearnLinearSVC("lsvc", ["fit", "score"], max_iter=4000)
        node.set_input_value("fit_dataset", xtrain)
        node.set_input_value("fit_targets", ytrain)
        node.set_input_value("score_dataset", xtest)
        node.set_input_value("score_targets", ytest)

        node.execute()

        print(node.scores)


class TestSklearnPCA:
    def test_execution(self):
        X, y = load_iris(return_X_y=True, as_frame=True)

        node = ss.SklearnPCA("pca", ["fit", "transform"], n_components=3)
        node.set_input_value("fit_dataset", X)
        node.set_input_value("transform_dataset", X)

        node.execute()

        print(node.transformed_dataset)
