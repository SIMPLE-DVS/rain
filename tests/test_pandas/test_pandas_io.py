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

from sklearn.datasets import load_iris

from rain import PandasCSVLoader, PandasCSVWriter
from tests.test_commons import check_param_not_found


class TestPandasCSVLoader:
    def test_parameter_not_found_exception(self):
        """Tests whether the class raises a ParameterNotFound exception."""
        check_param_not_found(PandasCSVLoader, path="", x=8)

    def test_dataset_load(self, tmpdir):
        """Tests the execution of the PandasCSVLoader."""
        iris = load_iris(as_frame=True).data
        tmpcsv = tmpdir / "tmp_iris.csv"
        iris.to_csv(tmpcsv, index=False)

        pandas_loader = PandasCSVLoader("s1", path=tmpcsv.__str__())
        pandas_loader.execute()
        iris_loaded = pandas_loader.dataset

        assert iris.equals(iris_loaded)


class TestPandasCSVWriter:
    def test_parameter_not_found_exception(self):
        """Tests whether the class raises a ParameterNotFound exception."""
        check_param_not_found(PandasCSVLoader, path="", x=8)

    def test_dataset_write(self, tmpdir):
        """Tests the execution of the PandasCSVWriter."""
        iris = load_iris(as_frame=True).data
        tmpcsv = tmpdir / "tmp_iris.csv"

        assert not tmpcsv.exists()

        pandas_writer = PandasCSVWriter(
            "s1", path=tmpcsv.__str__(), include_rows=False, delim=";"
        )
        pandas_writer.dataset = iris
        pandas_writer.execute()

        assert tmpcsv.exists()
