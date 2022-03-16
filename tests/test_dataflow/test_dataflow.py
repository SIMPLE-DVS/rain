import pytest

from rain import (
    DataFlow,
    PandasCSVLoader,
    PandasPivot,
    PandasRenameColumn,
    IrisDatasetLoader,
)
from rain.core.exception import DuplicatedNodeId, CyclicDataFlowException


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

    def test_get_node(self, dataflow):
        n = PandasCSVLoader("load", "./iris.csv")
        dataflow.add_node(n)
        assert dataflow.get_node("load") == n
        assert dataflow.get_node("piv") is None

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
        dataflow.add_edge(n @ "dataset" > t @ "dataset")

        assert (
            dataflow.has_node(n)
            and dataflow.has_node("piv")
            and "dataset" in dataflow.get_edge(n, t).source.nodes_attributes
        )

    def test_add_edges(self, dataflow):
        n = PandasCSVLoader("load", "./iris.csv")
        t = PandasPivot("piv", "r", "c", "v")
        r = PandasRenameColumn("rcol", [])
        dataflow.add_edges(
            [n @ "dataset" > t @ "dataset", n @ "dataset" > r @ "dataset"]
        )

        assert (
            dataflow.has_node(n)
            and dataflow.has_node(t)
            and dataflow.has_node(r)
            and "dataset" in dataflow.get_edge(n, t).source.nodes_attributes
            and "dataset" in dataflow.get_edge(n, r).source.nodes_attributes
        )

    def test_is_acyclic(self, dataflow):
        n = PandasCSVLoader("load", "./iris.csv")
        t = PandasPivot("piv", "r", "c", "v")
        dataflow.add_nodes([n, t])
        dataflow.add_edge(n @ "dataset" > t @ "dataset")
        assert dataflow.is_acyclic()

    def test_not_acyclic(self, dataflow):
        n = PandasCSVLoader("load", "./iris.csv")
        t = PandasPivot("piv", "r", "c", "v")
        r = PandasRenameColumn("rcol", [])
        dataflow.add_nodes([n, t])
        dataflow.add_edges(
            [
                n @ "dataset" > t @ "dataset",
                t @ "dataset" > r @ "dataset",
                r @ "dataset" > t @ "dataset",
            ]
        )
        assert not dataflow.is_acyclic()

    def test_get_execution_ordered_nodes(self, dataflow):
        n = PandasCSVLoader("load", "./iris.csv")
        t = PandasPivot("piv", "r", "c", "v")
        r = PandasRenameColumn("rcol", [])
        dataflow.add_edges(
            [
                n @ "dataset" > t @ "dataset",
                n @ "dataset" > r @ "dataset",
                t @ "dataset" > r @ "dataset",
            ]
        )
        assert dataflow.get_execution_ordered_nodes() == [n, t, r]

    def test_execution(self):
        df = DataFlow("dataflow1")
        load = IrisDatasetLoader("iris")
        rename = PandasRenameColumn(
            "rcol",
            columns=[
                "lungh. sepalo",
                "largh. sepalo",
                "lungh. petalo",
                "largh. petalo",
            ],
        )

        df.add_edge(load @ "dataset" > rename @ "dataset")
        df.execute()

        assert list(rename.dataset.columns) == [
            "lungh. sepalo",
            "largh. sepalo",
            "lungh. petalo",
            "largh. petalo",
        ]

    def test_execution_cyclic(self):
        df = DataFlow("dataflow1")
        load = IrisDatasetLoader("iris")
        rename = PandasRenameColumn(
            "rcol",
            columns=[
                "lungh. sepalo",
                "largh. sepalo",
                "lungh. petalo",
                "largh. petalo",
            ],
        )
        pivot = PandasPivot("piv", "lungh. sepalo", "largh. sepalo", "lungh. petalo")

        df.add_edges(
            [
                load @ "dataset" > rename @ "dataset",
                rename @ "dataset" > pivot @ "dataset",
                pivot @ "dataset" > rename @ "dataset",
            ]
        )
        with pytest.raises(CyclicDataFlowException):
            df.execute()
