import pytest
from sklearn.datasets import load_iris

from simple_repo.exception import ParameterNotFound
from simple_repo.simple_pandas.load_nodes import PandasCSVLoader, PandasCSVWriter


def check_class_integrity(class_):
    """Checks whether the class has all the needed class attributes."""
    assert hasattr(class_, "_input_vars")
    assert hasattr(class_, "_parameters")
    assert hasattr(class_, "_output_vars")


def check_instantiation_integrity(obj):
    """Checks after instantiation if the object has all the attributes described in its class attributes."""
    assert all(hasattr(obj, element) for element in obj._input_vars)
    assert all(hasattr(obj, element) for element in obj._output_vars)


def check_parents_integrity(class_: object, parent_vars: dict):
    """Checks that after instantiation all of its parents have not been modified."""
    if hasattr(class_, "__bases__") and class_.__bases__:
        for base in class_.__bases__:
            old_base_vars = [*parent_vars.get(base).keys()]
            new_base_vars = [*get_parents_vars(class_).get(base).keys()]

            assert all(k in old_base_vars for k in new_base_vars)


def get_parents_vars(class_):
    """Gets all the input, output and parameters vars of any parent of the class up to object (excluded)."""
    parents = {}
    if (
        hasattr(class_, "__bases__")
        and class_.__bases__
        and object not in class_.__bases__
    ):
        for base in class_.__bases__:
            base_vars = {}
            if hasattr(base, "_input_vars"):
                base_vars["_input_vars"] = base._input_vars
            if hasattr(base, "_parameters"):
                base_vars["_parameters"] = base._parameters
            if hasattr(base, "_output_vars"):
                base_vars["_output_vars"] = base._output_vars
            parents[base] = base_vars
            parents.update(get_parents_vars(base))
    return parents


def check_type_error(class_, **kwargs):
    """Checks whether the class raises a TypeError exception."""
    with pytest.raises(TypeError):
        class_(**kwargs)


def check_param_not_found(class_, **kwargs):
    """Checks whether the class raises a ParameterNotFound exception."""
    with pytest.raises(ParameterNotFound):
        class_(**kwargs)


class TestPandasCSVLoader:
    def test_instantiation(self):
        """Tests the integrity of the class and its parents before and after instantiation."""
        check_class_integrity(PandasCSVLoader)
        parents_vars = get_parents_vars(PandasCSVLoader)
        check_instantiation_integrity(PandasCSVLoader(path=""))
        check_parents_integrity(PandasCSVLoader, parents_vars)

    def test_parameter_type_exception(self):
        """Tests whether the class raises a TypeError exception."""
        check_type_error(PandasCSVLoader, path=5, delimiter=8)

    def test_parameter_not_found_exception(self):
        """Tests whether the class raises a ParameterNotFound exception."""
        check_param_not_found(PandasCSVLoader, path="", x=8)

    def test_dataset_load(self, tmpdir):
        """Tests the execution of the PandasCSVLoader."""
        iris = load_iris(as_frame=True).data
        tmpcsv = tmpdir / "tmp_iris.csv"
        iris.to_csv(tmpcsv, index=False)

        pandas_loader = PandasCSVLoader(path=tmpcsv.__str__())
        pandas_loader.execute()
        iris_loaded = pandas_loader.dataset

        assert iris.equals(iris_loaded)


class TestPandasCSVWriter:
    def test_instantiation(self):
        """Tests the integrity of the class and its parents before and after instantiation."""
        check_class_integrity(PandasCSVLoader)
        parents_vars = get_parents_vars(PandasCSVLoader)
        check_instantiation_integrity(PandasCSVLoader(path=""))
        check_parents_integrity(PandasCSVLoader, parents_vars)

    def test_parameter_type_exception(self):
        """Tests whether the class raises a TypeError exception."""
        check_type_error(PandasCSVLoader, path=5, delimiter=8)

    def test_parameter_not_found_exception(self):
        """Tests whether the class raises a ParameterNotFound exception."""
        check_param_not_found(PandasCSVLoader, path="", x=8)

    def test_dataset_write(self, tmpdir):
        """Tests the execution of the PandasCSVWriter."""
        iris = load_iris(as_frame=True).data
        tmpcsv = tmpdir / "tmp_iris.csv"

        assert not tmpcsv.exists()

        pandas_writer = PandasCSVWriter(
            path=tmpcsv.__str__(), include_rows=False, delim=";"
        )
        pandas_writer.dataset = iris
        pandas_writer.execute()

        assert tmpcsv.exists()
