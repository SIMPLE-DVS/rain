import pytest

from simple_repo import DataFlow, PandasCSVLoader, PandasPivot
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

    def test_add_edges(self):
        pass

    def test_is_acyclic(self, dataflow):
        n = PandasCSVLoader("load", "./iris.csv")
        t = PandasPivot("piv", "r", "c", "v")
        dataflow.add_nodes([n, t])
        assert dataflow.is_acyclic()

    def test_execution(self):
        pass
