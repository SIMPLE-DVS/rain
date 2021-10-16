import pytest
from sklearn.datasets import load_iris

from simple_repo import (
    PandasColumnSelector,
    PandasPivot,
    PandasRenameColumn,
)
from simple_repo.exception import ParameterNotFound
from tests.test_commons import check_param_not_found


@pytest.fixture
def iris_data():
    yield load_iris(as_frame=True).data


class TestPandasColumnSelector:
    @pytest.fixture
    def selector_data(self):
        correct_cols = [{"name": "sepal length (cm)", "type": "float"}]
        selector = PandasColumnSelector("pcs1", columns=correct_cols)
        yield selector

    def test_parameter_not_found_exception(self):
        """Tests whether the class raises a ParameterNotFound exception."""
        incorrect_cols = [{"nome": "c1", "type": "tipo"}]
        with pytest.raises(ParameterNotFound):
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
    pass


class TestPandasAddColumn:
    pass
