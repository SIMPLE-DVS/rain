import pandas as pd
import numpy as np
import pytest
from sklearn.datasets import load_iris

from simple_repo import (
    PandasColumnsFiltering,
    PandasSequence,
    PandasRenameColumn,
)
from simple_repo.exception import ParametersException, PandasSequenceException
from simple_repo.simple_pandas.transform_nodes import PandasSelectRows, PandasFilterRows


@pytest.fixture
def iris_data():
    yield load_iris(as_frame=True).data


@pytest.fixture
def initial_dataframe():
    yield pd.DataFrame(
        np.array(([1, 2, 3], [4, 5, 6])),
        index=["mouse", "rabbit"],
        columns=["one", "two", "three"],
    )


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

    def test_columns_type_str(self, initial_dataframe):
        prf = PandasColumnsFiltering("prf", columns_type="int")
        prf.set_input_value("dataset", initial_dataframe)

        prf.execute()

        expected_df = initial_dataframe.astype("int")

        assert prf.dataset.equals(expected_df)

    def test_columns_type_list(self, initial_dataframe):
        prf = PandasColumnsFiltering("prf", columns_type=["int", "float", "int"])
        prf.set_input_value("dataset", initial_dataframe)

        prf.execute()

        assert (
            prf.dataset["one"].dtype == "int32"
            and prf.dataset["two"].dtype == "float64"
            and prf.dataset["three"].dtype == "int32"
        )

    def test_columns_type_list_none(self, initial_dataframe):
        prf = PandasColumnsFiltering("prf", columns_type=["int", None, "int"])
        prf.set_input_value("dataset", initial_dataframe)

        prf.execute()

        assert (
            prf.dataset["one"].dtype == "int32"
            and prf.dataset["three"].dtype == "int32"
        )


class TestPandasSelectRows:
    def test_execution(self):
        df = pd.DataFrame(
            [range(3), [0, np.NaN, 0], [0, 0, np.NaN], range(3), range(3)]
        )

        prf = PandasSelectRows("selrows", select_nan=True)
        prf.set_input_value("dataset", df)

        prf.execute()

        expected_df = pd.Series([False, True, True, False, False])

        assert prf.selection.equals(expected_df)


class TestPandasFilterRows:
    def test_execution(self):
        df = pd.DataFrame(
            [range(3), [0, np.NaN, 0], [0, 0, np.NaN], range(3), range(3)]
        )

        select = PandasSelectRows("selrows", select_nan=True)
        select.set_input_value("dataset", df)

        select.execute()

        filter = PandasFilterRows("filtrows")
        filter.set_input_value("dataset", df)
        filter.set_input_value("selected_rows", select.selection)

        filter.execute()

        assert filter.dataset.isnull().iloc[0, 1] and filter.dataset.isnull().iloc[1, 2]


class TestPandasPivot:
    pass


class TestPandasAddColumn:
    pass


class TestPandasSequence:
    def test_exception_contains_non_computational_node(self):
        from simple_repo import PandasIrisLoader

        with pytest.raises(PandasSequenceException):
            PandasSequence(
                "ps",
                stages=[
                    PandasIrisLoader("pil"),
                    PandasColumnsFiltering("pcf", columns_range=(0, 1)),
                ],
            )

    # def test_exception_using_non_pandas_stages(self):
    #     # TODO AttributeError: 'SimpleKMeans' object has no attribute '_get_params_as_dict()'. Fixare quest'errore prima di testare questo.
    #     from simple_repo import SimpleKMeans
    #     with pytest.raises(PandasSequenceException):
    #         ps = PandasSequence("ps", stages=[
    #             PandasColumnsFiltering("pcf", columns_range=(0, 1)),
    #             SimpleKMeans("skm", ["fit"])
    #         ])

    def test_execution(self, initial_dataframe):
        ps = PandasSequence(
            "ps4",
            stages=[
                PandasRenameColumn("prc", columns=["a", "b", "c"]),
                PandasColumnsFiltering("pcf", column_names=["a", "c"]),
            ],
        )

        ps.set_input_value("dataset", initial_dataframe)

        ps.execute()

        expected_df = pd.DataFrame(
            np.array(([1, 3], [4, 6])),
            index=["mouse", "rabbit"],
            columns=["a", "c"],
        )

        assert ps.dataset.equals(expected_df)

    def test_integration_execution(self, tmpdir, iris_data):
        # setup input dataset
        iris_file = tmpdir / "iris.csv"

        iris_data.to_csv(iris_file, index=False)

        # setup sequence w/ stages
        import simple_repo as sr

        df = sr.DataFlow("df1")

        load = sr.PandasCSVLoader("loader", iris_file)
        ps = PandasSequence(
            "ps4",
            stages=[
                PandasRenameColumn("prc", columns=["a", "b", "c", "d"]),
                PandasColumnsFiltering("pcf", column_names=["a", "c"]),
            ],
        )

        df.add_edge(load > ps)

        df.execute()

        expected_df = iris_data.filter(
            axis=1, items=["sepal length (cm)", "petal length (cm)"]
        )
        expected_df.columns = ["a", "c"]

        assert ps.dataset.equals(expected_df)
