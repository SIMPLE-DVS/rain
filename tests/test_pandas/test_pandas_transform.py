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

import numpy
import pandas as pd
import numpy as np
import pytest
from sklearn.datasets import load_iris

from rain import (
    PandasColumnsFiltering,
    PandasSequence,
    PandasRenameColumn,
    ZScoreTrainer,
    ZScorePredictor,
)
from rain.core.exception import ParametersException, PandasSequenceException
from rain.nodes.pandas.transform_nodes import (
    PandasSelectRows,
    PandasFilterRows,
    PandasDropNan,
    PandasAddColumn,
    PandasReplaceColumn,
    PandasPivot,
    PandasGroupBy,
)


@pytest.fixture
def iris():
    yield load_iris(as_frame=True)


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
            (prf.dataset["one"].dtype == "int32" or prf.dataset["one"].dtype == "int64")
            and prf.dataset["two"].dtype == "float64"
            and (prf.dataset["three"].dtype == "int32" or prf.dataset["three"].dtype == "int64")
        )

    def test_columns_type_list_none(self, initial_dataframe):
        prf = PandasColumnsFiltering("prf", columns_type=["int", None, "int"])
        prf.set_input_value("dataset", initial_dataframe)

        prf.execute()

        assert (
            (prf.dataset["one"].dtype == "int32" or prf.dataset["one"].dtype == "int64")
            and (prf.dataset["three"].dtype == "int32" or prf.dataset["three"].dtype == "int64")
        )


class TestPandasSelectRows:
    def test_select_nan(self):
        df = pd.DataFrame(
            [range(3), [0, np.NaN, 0], [0, 0, np.NaN], range(3), range(3)]
        )

        prf = PandasSelectRows("selrows", select_nan=True)
        prf.set_input_value("dataset", df)

        prf.execute()

        expected_df = pd.Series([False, True, True, False, False])

        assert prf.selection.equals(expected_df)

    def test_condition(self):
        df = pd.DataFrame(
            [range(3), range(2, 7, 2), range(7, 10), range(3), range(12, 15)]
        )
        df.columns = ["one", "two", "three"]

        prf = PandasSelectRows("selrows", conditions=["three == 2 & one < 9"])
        prf.set_input_value("dataset", df)

        prf.execute()

        expected_df = pd.Series([True, False, False, True, False])

        assert prf.selection.equals(expected_df)

    def test_conditions(self):
        df = pd.DataFrame(
            [range(3), range(2, 7, 2), range(7, 10), range(3), range(12, 15)]
        )
        df.columns = ["one", "two", "three"]

        prf = PandasSelectRows(
            "selrows", conditions=["three == 2 & one < 9", "two >= 4 & two < 13"]
        )
        prf.set_input_value("dataset", df)

        prf.execute()

        expected_df = pd.Series([True, True, True, True, False])

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


class TestPandasDropNan:
    def test_execution(self):
        df = pd.DataFrame(
            [range(3), [0, np.NaN, 0], [0, 0, np.NaN], range(3), range(3)]
        )

        drop = PandasDropNan("dropnan", axis="rows", how="any")
        drop.set_input_value("dataset", df)

        drop.execute()

        has_nan = drop.dataset.isnull().values.any()

        assert not has_nan

    def test_invalid_axis(self):
        with pytest.raises(AttributeError):
            PandasDropNan("dropnan", axis="index", how="any")


class TestPandasPivot:
    def test_pivot(self, iris):
        ac = PandasPivot("ac", rows="sepal length (cm)", columns="sepal width (cm)", values="petal length (cm)")
        ac.dataset = iris.data
        r = ac.dataset["sepal length (cm)"].unique().size
        c = ac.dataset["sepal width (cm)"].unique().size
        assert len(ac.dataset.columns) == 4
        ac.execute()
        assert ac.dataset.shape == (r, c)


class TestPandasAddColumn:
    def test_add_column(self, iris):
        ac = PandasAddColumn("ac", 2, "prova")
        ac.dataset = iris.data
        ac.column = pd.Series(range(0, 150))
        assert len(ac.dataset.columns) == 4
        ac.execute()
        assert len(ac.dataset.columns) == 5
        assert "prova" in ac.dataset.columns
        assert ac.dataset["prova"].equals(pd.Series(range(0, 150)))


class TestPandasReplaceColumn:
    def test_replace_column(
        self,
    ):
        ac = PandasReplaceColumn("rc", 10, 11)
        ac.column = pd.Series([True, False, False, False, True])
        ac.execute()
        assert len(ac.column) == 5
        assert numpy.array_equal(
            ac.column.values, pd.Series([10, 11, 11, 11, 10]).values
        )


class TestPandasGroupBy:
    def test_group_by(self, iris):
        ac = PandasGroupBy("ac", key="time", freq="2D")
        from datetime import datetime, timedelta
        from random import randint
        start = datetime(2000, 1, 1)
        end = start + timedelta(days=iris.frame.shape[0])
        datetime_column = [str(start + timedelta(days=x)) for x in range(0, (end - start).days)]
        timedelta_column = [str(timedelta(hours=randint(0, 3), minutes=randint(0, 59))) for _ in
                            range(iris.frame.shape[0])]
        iris.frame.insert(value=datetime_column, loc=iris.frame.shape[1], column="time")
        iris.frame.insert(value=timedelta_column, loc=iris.frame.shape[1], column="delta")
        assert len(iris.frame.columns) == 7
        pcf = PandasColumnsFiltering("pcf", columns_type=["float64", "float64", "float64", "float64", "int32",
                                                          "datetime", "timedelta"])
        pcf.dataset = iris.frame
        pcf.execute()
        ac.dataset = pcf.dataset
        rows = ac.dataset.shape[0]
        ac.execute()
        assert ac.dataset.shape[0] == rows / 2
        assert len(ac.dataset.columns) == 6


class TestPandasSequence:
    def test_exception_contains_non_computational_node(self):
        from rain import IrisDatasetLoader

        with pytest.raises(PandasSequenceException):
            PandasSequence(
                "ps",
                stages=[
                    IrisDatasetLoader("pil"),
                    PandasColumnsFiltering("pcf", columns_range=(0, 1)),
                ],
            )

    # def test_exception_using_non_pandas_stages(self):
    #     # TODO AttributeError: 'SimpleKMeans' object has no attribute '_get_params_as_dict()'. Fixare quest'errore prima di testare questo.
    #     from rain import SimpleKMeans
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

    def test_integration_execution(self, tmpdir, iris):
        # setup input dataset
        iris_file = tmpdir / "iris.csv"

        iris.data.to_csv(iris_file, index=False)

        # setup sequence w/ stages
        import rain as sr

        df = sr.DataFlow("df1")

        load = sr.PandasCSVLoader("loader", iris_file)
        ps = PandasSequence(
            "ps4",
            stages=[
                PandasRenameColumn("prc", columns=["a", "b", "c", "d"]),
                PandasColumnsFiltering("pcf", column_names=["a", "c"]),
            ],
        )

        df.add_edge(load @ "dataset" > ps @ "dataset")

        df.execute()

        expected_df = iris.data.filter(
            axis=1, items=["sepal length (cm)", "petal length (cm)"]
        )
        expected_df.columns = ["a", "c"]

        assert ps.dataset.equals(expected_df)


class TestZScore:
    def test_zscore(self, iris):
        zscore_trainer = ZScoreTrainer("zt")
        zscore_trainer.dataset = iris.frame
        zscore_trainer.execute()
        model = zscore_trainer.model
        assert model is not None
        zscore_predictor = ZScorePredictor("zp")
        zscore_predictor.dataset = iris.frame
        zscore_predictor.model = model
        zscore_predictor.execute()
        assert zscore_predictor.predictions is not None
