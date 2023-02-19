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
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import rain.nodes.sklearn as ss
from rain.nodes.sklearn.node_structure import SklearnNode

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

        print(node.score_value)
        print(node.labels)


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

        print(node.score_value)


class TestSklearnPCA:
    def test_execution(self):
        X, y = load_iris(return_X_y=True, as_frame=True)

        node = ss.SklearnPCA("pca", ["fit", "transform"], n_components=3)
        node.set_input_value("fit_dataset", X)
        node.set_input_value("transform_dataset", X)

        node.execute()

        print(node.transformed_dataset)
