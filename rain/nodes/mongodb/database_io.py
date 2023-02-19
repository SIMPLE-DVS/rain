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

import ssl

import pandas
import pandas as pd
import pymongo

from rain.core.base import InputNode, OutputNode, Tags, LibTag, TypeTag
from rain.core.parameter import Parameters, KeyValueParameter


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

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.MONGODB, TypeTag.OUTPUT)


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

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.MONGODB, TypeTag.INPUT)
