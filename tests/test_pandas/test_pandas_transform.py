import pandas as pd
import numpy as np
import pytest
from sklearn.datasets import load_iris

from simple_repo import (
    PandasColumnsFiltering,
)
from simple_repo.exception import ParametersException


@pytest.fixture
def iris_data():
    yield load_iris(as_frame=True).data


incorrect_parameters = [
    {"column_indexes": [1, 2], "column_names": ["one", "two"]},
    {"column_indexes": [1, 2], "columns_like": "one"},
    {"column_names": ["one", "two"], "columns_regex": "e$"},
    {"columns_range": (0, 3), "columns_regex": "e$"},
]


class TestPandasColumnsFiltering:
    @pytest.mark.parametrize("params", incorrect_parameters)
    def test_parameter_exception(self, params):
        with pytest.raises(ParametersException):
            PandasColumnsFiltering("prf", **params)

    @pytest.fixture
    def initial_dataframe(self):
        yield pd.DataFrame(
            np.array(([1, 2, 3], [4, 5, 6])),
            index=["mouse", "rabbit"],
            columns=["one", "two", "three"],
        )

    def test_column_index_filtering(self, initial_dataframe):
        prf = PandasColumnsFiltering("prf", column_indexes=[0])
        prf.set_input_value("dataset", initial_dataframe)

        prf.execute()

        expected_df = pd.DataFrame(
            np.array(([1], [4])), index=["mouse", "rabbit"], columns=["one"]
        )

        assert prf.dataset.equals(expected_df)

    def test_column_index_range_filtering(self, initial_dataframe):
        prf = PandasColumnsFiltering("prf", columns_range=(0, 2))
        prf.set_input_value("dataset", initial_dataframe)

        prf.execute()

        expected_df = pd.DataFrame(
            np.array(([1, 2], [4, 5])),
            index=["mouse", "rabbit"],
            columns=["one", "two"],
        )

        assert prf.dataset.equals(expected_df)

    def test_column_names_filtering(self, initial_dataframe):
        prf = PandasColumnsFiltering("prf", column_names=["one", "three"])
        prf.set_input_value("dataset", initial_dataframe)

        prf.execute()

        expected_df = pd.DataFrame(
            np.array(([1, 3], [4, 6])),
            index=["mouse", "rabbit"],
            columns=["one", "three"],
        )

        assert prf.dataset.equals(expected_df)

    def test_column_regex_filtering(self, initial_dataframe):
        prf = PandasColumnsFiltering("prf", columns_regex="e$")
        prf.set_input_value("dataset", initial_dataframe)

        prf.execute()

        expected_df = pd.DataFrame(
            np.array(([1, 3], [4, 6])),
            index=["mouse", "rabbit"],
            columns=["one", "three"],
        )

        assert prf.dataset.equals(expected_df)

    def test_column_like_filtering(self, initial_dataframe):
        prf = PandasColumnsFiltering("prf", columns_like="o")
        prf.set_input_value("dataset", initial_dataframe)

        prf.execute()

        expected_df = pd.DataFrame(
            np.array(([1, 2], [4, 5])),
            index=["mouse", "rabbit"],
            columns=["one", "two"],
        )

        assert prf.dataset.equals(expected_df)


class TestPandasPivot:
    pass


class TestPandasAddColumn:
    pass
