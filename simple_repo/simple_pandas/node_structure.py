from abc import abstractmethod
from collections import OrderedDict

import pandas as pd
from typing import List

from simple_repo.base import SimpleNode, Singleton, Node, get_class
from simple_repo.parameter import SimpleIO


def reset(simple_node):
    dic = vars(simple_node)
    for i in dic.keys():
        dic[i] = None


class PandasNode(SimpleNode):
    """
    Every PandasNode takes a pandas DataFrame as input,
    applies a tansformation and returns a pandas DataFrame as output.
    """

    _input_vars = {"dataset": SimpleIO(pd.DataFrame)}
    _parameters = {}
    _output_vars = {"dataset": pd.DataFrame}

    def __init__(self, **kwargs):
        super(PandasNode, self).__init__(**kwargs)

    @abstractmethod
    def execute(self):
        pass

    def __str__(self):
        return "{}".format(self._get_params_as_dict())


class PandasExecutor(metaclass=Singleton):
    def __init__(self):
        pass

    @staticmethod
    def convert(nodes: List[Node]):
        simple_nodes = OrderedDict()
        nodes_nexts = {}

        # carico le istanze dei SimpleNode a partire dai Node mantenendo l'ordinamento
        # mi tengo da parte anche le coppie id-then
        for node in nodes:
            node_class = get_class(node.node)

            s_node = node_class(**node.parameters)

            simple_nodes[node.node_id] = s_node

            if node.then:
                nodes_nexts[node.node_id] = node.then

        return simple_nodes, nodes_nexts

    @staticmethod
    def execute(simple_node: SimpleNode):
        simple_node.execute()
