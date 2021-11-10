import pytest

from simple_repo import PandasCSVLoader, PandasPivot, PandasRenameColumn
from simple_repo.exception import EdgeConnectionError


class TestEdgesConnections:
    def test_gt_node_node(self):
        n = PandasCSVLoader("load", "./iris.csv")
        t = PandasPivot("piv", "r", "c", "v")

        edge = n > t

        intersection = set(n._output_vars.keys()).intersection(
            set(t._input_vars.keys())
        )

        assert (
            edge.source == [n]
            and edge.destination == [t]
            and all(key in n._output_vars.keys() for key in intersection)
            and all(key in t._input_vars.keys() for key in intersection)
        )

    def test_sub_node_str(self):
        n = PandasCSVLoader("load", "./iris.csv")

        edge = n @ "dataset"

        assert (
            edge.source == [n]
            and edge.destination is None
            and type(edge.source_output) == list
            and "dataset" in edge.source_output
        )

    def test_sub_node_str_list(self):
        n = PandasCSVLoader("load", "./iris.csv")

        edge = n @ ["dataset1", "dataset2"]

        assert (
            edge.source == [n]
            and edge.destination is None
            and type(edge.source_output) == list
            and edge.source_output == ["dataset1", "dataset2"]
        )

    def test_sub_non_output_node(self):
        from simple_repo.simple_io import SparkSaveModel

        n = SparkSaveModel("load", "./iris.csv")

        with pytest.raises(EdgeConnectionError):
            n @ "dataset1"

    def test_sub_node_to_non_str_nor_list(self):
        n = PandasCSVLoader("load", "./iris.csv")

        with pytest.raises(EdgeConnectionError):
            n @ 5

        with pytest.raises(EdgeConnectionError):
            n @ ("dataset1", "dataset2")

    def test_sub_node_to_non_existing_var(self):
        n = PandasCSVLoader("load", "./iris.csv")

        with pytest.raises(EdgeConnectionError):
            n @ "non_existing_var"

    def test_gt_edge_to_destination(self):
        n = PandasCSVLoader("load", "./iris.csv")
        t = PandasPivot("piv", "r", "c", "v")

        edge = n @ "dataset" > t

        assert (
            edge.source == [n]
            and edge.destination == [t]
            and "dataset" in edge.source_output
            and "dataset" in edge.destination_input
        )

    def test_gt_edge_list_to_destination(self):
        n = PandasCSVLoader("load", "./iris.csv")
        t = PandasPivot("piv", "r", "c", "v")

        edge = n @ ["dataset1", "dataset2"] > t

        assert (
            edge.source == [n]
            and edge.destination == [t]
            and ["dataset1", "dataset2"] == edge.source_output
            and ["dataset1", "dataset2"] == edge.destination_input
        )

    def test_gt_edge_at_to_destination(self):
        n = PandasCSVLoader("load", "./iris.csv")
        t = PandasPivot("piv", "r", "c", "v")

        edge = n @ "dataset" > t @ "dataset"

        assert (
            edge.source == [n]
            and edge.destination == [t]
            and "dataset" in edge.source_output
            and "dataset" in edge.destination_input
        )

    def test_gt_edge_at_to_destination_node_list(self):
        n = PandasCSVLoader("load", "./iris.csv")
        t = PandasPivot("piv", "r", "c", "v")
        r = PandasRenameColumn("rcol", [])

        edge = n @ "dataset" > t & r & n @ "dataset"

        assert (
            edge.source == [n]
            and edge.destination == [t, r, n]
            and "dataset" in edge.source_output
            and "dataset" in edge.destination_input
        )
