import pytest
from sklearn.datasets import load_iris

from simple_repo import MongoCSVReader, MongoCSVWriter


@pytest.fixture
def iris_data():
    yield load_iris(as_frame=True).data


class TestMongoCSVWriter:
    def test_mongo_writer(self, iris_data):
        w = MongoCSVWriter(
            "mongo_writer",
            "mongodb+srv://admin:admin@cluster0.yhcxc.mongodb.net/simple?retryWrites=true&w=majority",
            "simple",
            "temp_dataset",
        )
        w.dataset = iris_data
        collection = w.execute()
        items = list(collection.find())
        assert len(items) == 150
        assert len(list(items[0].keys())) == 5
        collection.drop()
        print("ok")


class TestMongoCSVReader:
    def test_mongo_reader(self):
        r = MongoCSVReader(
            "mongo_writer",
            "mongodb+srv://admin:admin@cluster0.yhcxc.mongodb.net/simple?retryWrites=true&w=majority",
            "simple",
            "dataset",
            projection={"_id": False},
        )
        assert r.dataset is None
        r.execute()
        assert r.dataset is not None
        assert "_id" not in r.dataset.columns
        assert len(r.dataset.columns) == 4
        assert len(r.dataset.index) == 150
