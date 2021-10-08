import pytest
from sklearn.datasets import load_iris

from simple_repo.exception import ParameterNotFound
from simple_repo import (
    PandasColumnSelector,
    PandasPivot,
    PandasRenameColumn,
)


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


@pytest.fixture
def iris_data():
    yield load_iris(as_frame=True).data


class TestPandasColumnSelector:
    @pytest.fixture
    def selector_data(self):
        correct_cols = [{"name": "sepal length (cm)", "type": "float"}]
        selector = PandasColumnSelector(columns=correct_cols)
        yield selector
        selector = None
        correct_cols = None

    def test_instantiation(self, selector_data):
        """Tests the integrity of the class and its parents before and after instantiation."""
        check_class_integrity(PandasColumnSelector)
        parents_vars = get_parents_vars(PandasColumnSelector)
        check_instantiation_integrity(selector_data)
        check_parents_integrity(PandasColumnSelector, parents_vars)

    def test_parameter_not_found_exception(self):
        """Tests whether the class raises a ParameterNotFound exception."""
        incorrect_cols = [{"nome": "c1", "type": "tipo"}]
        check_param_not_found(PandasColumnSelector, columns=incorrect_cols)

    def test_column_selection(self, iris_data, selector_data):
        sel = selector_data
        sel.dataset = iris_data
        print(sel.dataset)
        sel.execute()
        print(sel.dataset, sel.dataset.shape, sel.dataset.iloc[:, 0].dtype)
        assert sel.dataset.shape == (iris_data.shape[0], 1)
        assert sel.dataset.iloc[:, 0].dtype == float


class TestPandasPivot:
    def test_instantiation(self):
        """Tests the integrity of the class and its parents before and after instantiation."""
        check_class_integrity(PandasPivot)
        parents_vars = get_parents_vars(PandasPivot)
        check_instantiation_integrity(PandasPivot(rows=""))
        check_parents_integrity(PandasPivot, parents_vars)


class TestPandasAddColumn:
    def test_instantiation(self):
        """Tests the integrity of the class and its parents before and after instantiation."""
        check_class_integrity(PandasRenameColumn)
        parents_vars = get_parents_vars(PandasRenameColumn)
        check_instantiation_integrity(PandasRenameColumn(columns=[""]))
        check_parents_integrity(PandasRenameColumn, parents_vars)


if __name__ == "__main__":
    pass
    # import simple_repo.pandas
    # import simple_repo.sklearn
    # import simple_repo.executors
    #
    # nodo1 = pandas.CSVLoader(path=....) # load_csv
    # nodo2 = pandas.Filtering(select_columns=[...], filter_values=[...]) #
    # nodo3 = sklearn.LSVC(parametri, execute=['fit', 'predict'])
    # nodo4 =
    #
    # nodo1 >> nodo2
    # nodo2 >> [nodo3, nodo4]
    # nodo3 >> nodo4
    #
    #
    # dag = nodo1 >> nodo2 > (dataset, train_dataset) > nodo3
    #
    # executors.LocalExecutor(dag)
