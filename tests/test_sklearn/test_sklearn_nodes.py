"""
 Copyright (C) 2023 Università degli Studi di Camerino and Sigma S.p.A.
 Authors: Alessandro Antinori, Rosario Capparuccia, Riccardo Coltrinari, Flavio Corradini, Marco Piangerelli, Barbara Re, Marco Scarpetta

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU Affero General Public License as
 published by the Free Software Foundation, either version 3 of the
 License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Affero General Public License for more details.

 You should have received a copy of the GNU Affero General Public License
 along with this program.  If not, see <https://www.gnu.org/licenses/>.
 """

import pytest
import sklearn.base
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import rain.nodes.sklearn as ss
from rain.nodes.sklearn.node_structure import SklearnNode

sklearn_nodes = [ss.SklearnLinearSVC, ss.SimpleKMeans, ss.SklearnPCA]


@pytest.mark.parametrize("class_or_obj", sklearn_nodes)
def test_sklearn_methods(class_or_obj):
    """Checks whether the sklearn node has no method attributes."""
    assert issubclass(class_or_obj, SklearnNode) and hasattr(class_or_obj, "_methods")


@pytest.mark.parametrize("class_or_obj", sklearn_nodes)
def test_sklearn_execute(class_or_obj):
    """Checks whether the sklearn node raises an error when executed with a non-existing method."""
    with pytest.raises(Exception):
        class_or_obj("node", execute=["non_existing_method"])


@pytest.mark.parametrize("class_or_obj", sklearn_nodes)
def test_sklearn_no_dataset_but_model_fit(class_or_obj):
    iris = load_iris(as_frame=True).data
    kmeans = ss.SimpleKMeans("km", ["fit"])
    kmeans.set_input_value("dataset", iris)
    kmeans.execute()
    fitted_model = kmeans.fitted_model

    node = class_or_obj("node", execute=["fit"])
    node.set_input_value("fitted_model", fitted_model)
    node.execute()
    assert node.fitted_model is not None


@pytest.mark.parametrize("class_or_obj", sklearn_nodes)
def test_sklearn_model_from_dataset(class_or_obj):
    node = class_or_obj("node", execute=["fit"])
    X, y = load_iris(return_X_y=True, as_frame=True)
    node.set_input_value("dataset", X)
    if "fit_targets" in class_or_obj.__dict__:
        node.set_input_value("fit_targets", y)
    node.execute()
    assert node.fitted_model is not None


@pytest.mark.parametrize("class_or_obj", sklearn_nodes)
def test_sklearn_dont_overwrite_model(class_or_obj):
    node = class_or_obj("node", execute=["fit"])
    X, y = load_iris(return_X_y=True, as_frame=True)
    base_estimator = sklearn.base.BaseEstimator()
    base_estimator.labels_ = None  # To make the test pass
    node.set_input_value("fitted_model", base_estimator)
    node.set_input_value("dataset", X)
    if "fit_targets" in class_or_obj.__dict__:
        node.set_input_value("fit_targets", y)
    node.execute()
    assert node.fitted_model is base_estimator


class TestKMeans:
    def test_execution(self):
        iris = load_iris(as_frame=True).data
        iristrain, iristest = train_test_split(iris, test_size=0.15, shuffle=False)
        node = ss.SimpleKMeans("km", ["fit"])
        node.set_input_value("dataset", iristrain)

        node.execute()
        fitted_model = node.fitted_model
        assert fitted_model is not None

        node2 = ss.SimpleKMeans("km2", ["score"])
        node2.set_input_value("fitted_model", fitted_model)
        node2.set_input_value("dataset", iristest)

        node2.execute()
        assert node2.score_value is not None
        assert node2.labels is not None


class TestSklearnLinearSVC:
    def test_execution_passed_target(self):
        X, y = load_iris(return_X_y=True, as_frame=True)
        xtrain, xtest, ytrain, ytest = train_test_split(
            X, y, test_size=0.15, shuffle=False
        )

        node = ss.SklearnLinearSVC("lsvc", ["fit"], max_iter=4000)
        node.set_input_value("dataset", xtrain)
        node.set_input_value("fit_targets", ytrain)

        node.execute()
        fitted_model = node.fitted_model
        assert fitted_model is not None

        node2 = ss.SklearnLinearSVC("lsvc2", ["score", "predict"], max_iter=4000)
        node2.set_input_value("fitted_model", fitted_model)
        node2.set_input_value("dataset", xtest)
        node2.set_input_value("score_targets", ytest)

        node2.execute()
        assert node2.score_value is not None
        assert node2.predictions is not None


class TestSklearnPCA:
    def test_execution(self):
        X, y = load_iris(return_X_y=True, as_frame=True)

        node = ss.SklearnPCA("pca", ["fit"], n_components=3)
        node.set_input_value("dataset", X)

        node.execute()
        fitted_model = node.fitted_model
        assert fitted_model is not None

        node2 = ss.SklearnPCA("pca2", ["transform"], n_components=3)
        node2.set_input_value("fitted_model", fitted_model)
        node2.set_input_value("dataset", X)

        node2.execute()
        assert node2.transformed_dataset is not None


@pytest.mark.parametrize("class_or_obj", sklearn_nodes)
@pytest.mark.parametrize("method", ["fit", "predict", "score", "transform"])
def test_sklearn_no_dataset_fit_predict_score_transform(class_or_obj, method):
    """Checks whether the sklearn node raises an error when the 'fit', 'predict',
    'score' and 'transform' are executed with no input dataset."""
    if method in class_or_obj._methods:
        with pytest.raises(Exception):
            node = class_or_obj("node", execute=[method])
            node.execute()
