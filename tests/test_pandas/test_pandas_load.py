import pytest
from sklearn.datasets import load_iris

from simple_repo.exception import ParameterNotFound
from simple_repo.simple_pandas.load_nodes import PandasCSVLoader, PandasCSVWriter


def check_type_error(**kwargs):
    with pytest.raises(TypeError):
        PandasCSVLoader(**kwargs)


def check_param_not_found(**kwargs):
    with pytest.raises(ParameterNotFound):
        PandasCSVLoader(**kwargs)


def check_mandatory_param(**kwargs):
    with pytest.raises(TypeError):
        PandasCSVLoader(**kwargs)


class TestPandasCSVLoader:
    def test_parameter_type_exception(self):
        check_type_error(path=5, delimiter=8)

    def test_parameter_not_found_exception(self):
        check_param_not_found(path="", x=8)

    def test_dataset_load(self, tmpdir):
        iris = load_iris(as_frame=True).data
        tmpcsv = tmpdir / "tmp_iris.csv"
        iris.to_csv(tmpcsv, index=False)

        pandas_loader = PandasCSVLoader(path=tmpcsv.__str__())
        pandas_loader.execute()
        iris_loaded = pandas_loader.dataset

        assert iris.equals(iris_loaded)


class TestPandasCSVWriter:
    def test_parameter_type_exception(self):
        check_type_error(path=5, delimiter=8)

    def test_parameter_not_found_exception(self):
        check_param_not_found(path="", x=8)

    def test_dataset_write(self, tmpdir):
        iris = load_iris(as_frame=True).data
        tmpcsv = tmpdir / "tmp_iris.csv"

        assert not tmpcsv.exists()

        pandas_writer = PandasCSVWriter(
            path=tmpcsv.__str__(), include_rows=False, delim=";"
        )
        pandas_writer.dataset = iris
        pandas_writer.execute()

        assert tmpcsv.exists()
