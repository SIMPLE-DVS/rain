import ssl

import pandas
import pandas as pd
import pymongo

from simple_repo.base import InputNode, OutputNode
from simple_repo.parameter import Parameters, KeyValueParameter


class MongoCSVWriter(OutputNode):
    """Write a Pandas Dataframe into a MongoDB collection.

    Parameters
    ----------
    node_id : str
        The unique id of the node.
    connection : str
        Hostname or IP address or Unix domain socket path of a single MongoDB instance to connect to, or a mongodb URI
    db : str
        Name of the database to connect to.
    coll : str
        Name of the collection to connect to.
    """

    _input_vars = {"dataset": pd.DataFrame}

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
        return collection


class MongoCSVReader(InputNode):
    """Read a Pandas Dataframe from a MongoDB collection.

    Parameters
    ----------
    node_id : str
        The unique id of the node.
    connection : str
        Hostname or IP address or Unix domain socket path of a single MongoDB instance to connect to, or a mongodb URI
    db : str
        Name of the database to connect to.
    coll : str
        Name of the collection to connect to.
    filter : dict, default None
        A SON object specifying elements which must be present for a document to be included in the result set
    projection : dict, default None
        A dict to exclude fields from the result (e.g. projection={'_id': False})
    """

    _output_vars = {"dataset": pd.DataFrame}

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
