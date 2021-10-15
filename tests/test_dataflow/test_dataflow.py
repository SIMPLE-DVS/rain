import pytest

from simple_repo import (
    DataFlow,
    PandasCSVLoader,
    PandasPivot,
    PandasRenameColumn,
    PandasIrisLoader,
)
from simple_repo.exception import DuplicatedNodeId


@pytest.fixture
def dataflow():
    return DataFlow("dataflow1")


class TestDataflow:
    def test_add_node(self, dataflow):
        n = PandasCSVLoader("load", "./iris.csv")
        assert dataflow.add_node(n)

    def test_already_present_node(self, dataflow):
        n = PandasCSVLoader("load", "./iris.csv")
        dataflow.add_node(n)
        with pytest.raises(DuplicatedNodeId):
            dataflow.add_node(n)

    def test_add_nodes(self, dataflow):
        n = PandasCSVLoader("load", "./iris.csv")
        t = PandasPivot("piv", "r", "c", "v")
        assert dataflow.add_nodes([n, t])

    def test_already_present_nodes(self, dataflow):
        n = PandasCSVLoader("load", "./iris.csv")
        t = PandasPivot("piv", "r", "c", "v")
        dataflow.add_node(t)
        with pytest.raises(DuplicatedNodeId):
            dataflow.add_nodes([n, t])

    def test_add_edge(self, dataflow):
        n = PandasCSVLoader("load", "./iris.csv")
        t = PandasPivot("piv", "r", "c", "v")
        dataflow.add_edge(n @ "dataset" > t)

        assert (
            dataflow.has_node(n)
            and dataflow.has_node("piv")
            and "dataset" in dataflow.get_edge(n, t).source_output
        )

    def test_add_edges(self, dataflow):
        n = PandasCSVLoader("load", "./iris.csv")
        t = PandasPivot("piv", "r", "c", "v")
        r = PandasRenameColumn("rcol", [])
        dataflow.add_edges([n @ "dataset" > t, n @ "dataset" > r])

        assert (
            dataflow.has_node(n)
            and dataflow.has_node(t)
            and dataflow.has_node(r)
            and "dataset" in dataflow.get_edge(n, t).source_output
            and "dataset" in dataflow.get_edge(n, r).source_output
        )

    def test_is_acyclic(self, dataflow):
        n = PandasCSVLoader("load", "./iris.csv")
        t = PandasPivot("piv", "r", "c", "v")
        dataflow.add_nodes([n, t])
        dataflow.add_edge(n > t)
        assert dataflow.is_acyclic()

    def test_not_acyclic(self, dataflow):
        n = PandasCSVLoader("load", "./iris.csv")
        t = PandasPivot("piv", "r", "c", "v")
        r = PandasRenameColumn("rcol", [])
        dataflow.add_nodes([n, t])
        dataflow.add_edges([n > t, t > r, r > t])
        assert not dataflow.is_acyclic()

    def test_get_execution_ordered_nodes(self, dataflow):
        n = PandasCSVLoader("load", "./iris.csv")
        t = PandasPivot("piv", "r", "c", "v")
        r = PandasRenameColumn("rcol", [])
        dataflow.add_edges([n @ "dataset" > t & r, t > r])
        assert dataflow.get_execution_ordered_nodes() == [n, t, r]

    def test_execution(self):
        df = DataFlow("dataflow1")
        load = PandasIrisLoader("iris")
        rename = PandasRenameColumn(
            "rcol",
            columns=[
                "lungh. sepalo",
                "largh. sepalo",
                "lungh. petalo",
                "largh. petalo",
            ],
        )

        df.add_edge(load @ "dataset" > rename)
        df.execute()

        assert list(rename.dataset.columns) == [
            "lungh. sepalo",
            "largh. sepalo",
            "lungh. petalo",
            "largh. petalo",
        ]
