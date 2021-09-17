from abc import abstractmethod
from collections import OrderedDict
from typing import List
from typing import Any

from pyspark.sql import SparkSession

from simple_repo.base import Node
from simple_repo.base import Singleton
from simple_repo.base import get_class
from simple_repo.base import SimpleNode
from simple_repo.dag import DagCreator

import pickle


def reset(simple_node):
    dic = vars(simple_node)
    for i in dic.keys():
        dic[i] = None


class SimpleExecutor(metaclass=Singleton):
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def convert(nodes: List[Node]):
        pass

    @staticmethod
    def execute(simple_node: SimpleNode):
        simple_node.execute()


class PandasExecutor(SimpleExecutor):
    def __init__(self):
        super(PandasExecutor, self).__init__()

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


class SklearnExecutor(SimpleExecutor):
    def __init__(self):
        super(SklearnExecutor, self).__init__()

    @staticmethod
    def convert(nodes: List[Node]):
        simple_nodes = OrderedDict()
        nodes_nexts = {}

        # if len(nodes) == 0:
        #     return None

        # carico le istanze dei SimpleNode a partire dai Node mantenendo l'ordinamento
        # mi tengo da parte anche le coppie id-then
        for node in nodes:
            node_class = get_class(node.node)

            s_node = node_class(node.execute, **node.parameters)

            simple_nodes[node.node_id] = s_node

            if node.then:
                nodes_nexts[node.node_id] = node.then

        return simple_nodes, nodes_nexts


class SparkExecutor(SimpleExecutor):
    def __init__(self):
        super(SparkExecutor, self).__init__()
        self._spark = None

    def convert(self, nodes: List[Node]):
        self._spark = SparkSession.builder.getOrCreate()
        simple_nodes = OrderedDict()
        nodes_nexts = {}

        # if len(nodes) == 0:
        #     return None

        # carico le istanze dei SimpleNode a partire dai Node mantenendo l'ordinamento
        # mi tengo da parte anche le coppie id-then
        for node in nodes:
            cls = get_class(node.node)
            if cls == get_class(
                "simple_repo.simple_spark.pipeline.spark_pipeline.SparkPipelineNode"
            ):
                stages = []
                pipe = node.parameters.get("stages")
                for s in pipe:
                    c = get_class(s.get("name"))
                    stage = c(spark=self._spark, **s.get("param"))
                    stages.append(stage)
                simple_node = cls(spark=self._spark, lst=stages)
            else:
                simple_node = cls(spark=self._spark, **node.parameters)

            simple_nodes[node.node_id] = simple_node

            if node.then:
                nodes_nexts[node.node_id] = node.then

        return simple_nodes, nodes_nexts


class ExecutionResult:
    """Mantains subpipelines execution results."""

    def __init__(self, receiver_id: str, receiver_field: str, data: Any):
        self._receiver_id = receiver_id
        self._receiver_field = receiver_field
        self._data = data

    @property
    def receiver_id(self):
        return self._receiver_id

    @property
    def receiver_field(self):
        return self._receiver_field

    @property
    def data(self):
        return self._data


class SimpleSubPipeline:
    def __init__(self, s_type, nodes: List[Node], executor):
        self._type = s_type
        self._nodes = OrderedDict()
        self._nexts = {}
        self._executor = executor

        s_nodes, nxts = self._executor.convert(nodes)

        self._nodes.update(s_nodes)

        if nxts is not None:
            self._nexts.update(nxts)

    def execute_subpipeline(self):
        results = []

        # per ogni nodo della sottopipeline
        for node_name, node in self._nodes.items():
            # esegui il nodo
            self._executor.execute(node)

            # se il nodo non ha next prosegui
            if node_name not in self._nexts.keys():
                continue

            # altrimenti per ogni next del nodo
            for nxt in self._nexts.get(node_name):
                # prendo id del ricevente
                receiver_id = nxt.get("send_to")

                # prendo rispettivamente i field del nodo e del ricevente
                for k, v in nxt.items():
                    # il campo send_to non è considerato un field
                    if k == "send_to":
                        continue

                    # prendo l'output generato dal nodo
                    node_output = node.get_output_value(k)

                    # creo un result
                    result = ExecutionResult(receiver_id, v, node_output)

                    # provo ad aggiornare il nodo ricevente con l'output
                    is_received = self.update_node_data(result)

                    # se l'output non è stato ricevuto allora il
                    # ricevente non fa parte di questa sottopipeline
                    if not is_received:
                        # allora aggiungo il risultato alla lista da ritornare
                        results.append(result)

        return results

    def reset(self):
        for s_node in self._nodes.values():
            reset(s_node)

    def get_node(self, node_id: str):
        return self._nodes.get(node_id)

    def update_node_data(self, data: ExecutionResult):
        # prendo il nodo da aggiornare
        receiver = self.get_node(data.receiver_id)

        # se non è presente nella sottopipeline ritorno False
        # per indicare che l'aggiornamento non è andato a buon fine
        if receiver is None:
            return False

        # aggiorno il nodo
        receiver.set_input_value(data.receiver_field, data.data)

        return True

    @property
    def executor(self):
        return self._executor

    @property
    def nodes(self):
        return self._nodes


class SimplePipeline:
    def __init__(self):
        self._subpipelines: List[SimpleSubPipeline] = []
        self._nodes = OrderedDict()

    def add_subpipeline(self, subpipeline: SimpleSubPipeline):
        self._subpipelines.append(subpipeline)

        for node in subpipeline.nodes.keys():
            self._nodes[node] = subpipeline

    def execute(self):
        # per ogni sottopipeline
        for subpip in self._subpipelines:
            # eseguo la sottopipeline
            results = subpip.execute_subpipeline()

            # se non ha prodotto risultati da propagare allora proseguo
            if not results:
                continue

            # altrimenti per ogni risultato
            for result in results:
                # prendo la sottopipeline a cui appartiene il nodo ricevente
                next_sub_pipeline = self._nodes.get(result.receiver_id)
                # mando il risultato alla sottopipeline che aggiornerà il nodo
                next_sub_pipeline.update_node_data(result)

            # resetto le variabili dell'intera pipeline
            subpip.reset()

    @property
    def subpipelines(self):
        return self._subpipelines


if __name__ == "__main__":
    import os

    sjp = DagCreator()

    sjp.create_dag("pandas_sklearn.yaml")

    # sjp.show_dag()

    pipeline = sjp.get_sub_pipelines()

    pickle.dump(
        pipeline, open("C:/Users/{}/Desktop/pipe.pkl".format(os.getlogin()), "wb")
    )
