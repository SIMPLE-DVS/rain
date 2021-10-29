import ssl

import pandas
import pymongo

from simple_repo.parameter import Parameters, KeyValueParameter
from simple_repo.simple_io.pandas_io import PandasInputNode, PandasOutputNode


class MongoCSVWriter(PandasOutputNode):
    """Write a Pandas Dataframe into a MongoDB collection.

    Parameters
    ----------
    node_id: str
        The unique id of the node.
    connection: str
        Hostname or IP address or Unix domain socket path of a single MongoDB instance to connect to, or a mongodb URI
    db: str
        Name of the database to connect to.
    coll: str
        Name of the collection to connect to.
    """

    def __init__(self, node_id: str, connection: str, db: str, coll: str):
        self.parameters = Parameters(
            connection=KeyValueParameter("connection", str, connection),
            db=KeyValueParameter("db", str, db),
            coll=KeyValueParameter("coll", str, coll),
        )
        super(MongoCSVWriter, self).__init__(node_id)

    def execute(self):
        params = self.parameters.get_dict()
        client = pymongo.MongoClient(
            params.get("connection"), ssl=True, ssl_cert_reqs=ssl.CERT_NONE
        )
        collection = client[params.get("db")][params.get("coll")]
        collection.insert_many(self.dataset.to_dict("records"))


class MongoCSVReader(PandasInputNode):
    """Read a Pandas Dataframe from a MongoDB collection.

    Parameters
    ----------
    node_id: str
        The unique id of the node.
    connection: str
        Hostname or IP address or Unix domain socket path of a single MongoDB instance to connect to, or a mongodb URI
    db: str
        Name of the database to connect to.
    coll: str
        Name of the collection to connect to.
    filter: dict
        A SON object specifying elements which must be present for a document to be included in the result set
    projection: dict
        A dict to exclude fields from the result (e.g. projection={'_id': False})
    """

    def __init__(
        self,
        node_id: str,
        connection: str,
        db: str,
        coll: str,
        filter: dict = None,
        projection: dict = None,
    ):
        self.parameters = Parameters(
            connection=KeyValueParameter("connection", str, connection),
            db=KeyValueParameter("db", str, db),
            coll=KeyValueParameter("coll", str, coll),
            filter=KeyValueParameter("filter", dict, filter),
            projection=KeyValueParameter("projection", dict, projection),
        )
        super(MongoCSVReader, self).__init__(node_id)

    def execute(self):
        params = self.parameters.get_dict()
        client = pymongo.MongoClient(
            params.get("connection"), ssl=True, ssl_cert_reqs=ssl.CERT_NONE
        )
        collection = client[params.get("db")][params.get("coll")]
        self.dataset = pandas.DataFrame(
            list(
                collection.find(
                    filter=params.get("filter"), projection=params.get("projection")
                )
            )
        )


if __name__ == "__main__":
    # w = MongoCSVWriter("mongo_writer",
    #                    "mongodb+srv://admin:admin@cluster0.yhcxc.mongodb.net/simple?retryWrites=true&w=majority",
    #                    "simple", "dataset")
    # w.dataset = pandas.read_csv("C:/Users/Marco/Desktop/iris_ds.csv")
    # w.execute()

    r = MongoCSVReader(
        "mongo_writer",
        "mongodb+srv://admin:admin@cluster0.yhcxc.mongodb.net/simple?retryWrites=true&w=majority",
        "simple",
        "dataset",
    )
    r.execute()
    print("ok")
