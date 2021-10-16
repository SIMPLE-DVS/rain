from sklearn.datasets import load_iris

from simple_repo import PandasCSVLoader, PandasCSVWriter
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
