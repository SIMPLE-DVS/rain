import json
from collections import OrderedDict
from typing import List

from pyspark.sql import SparkSession

from simple_repo.dag import SimpleJSONParser
from simple_repo.simple_spark.spark_node import SparkNode, Transformer, Estimator
from simple_repo.base import get_class, reset, Singleton, Node, SimpleNode


class SparkExecutor(metaclass=Singleton):
    def __init__(self):
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

    @staticmethod
    def execute(simple_node: SimpleNode):
        simple_node.execute()


if __name__ == "__main__":
    sjp = SimpleJSONParser()

    sjp.parse_configuration("simple_spark/spark_conf.json")

    spark_nodes = sjp.get_sorted_nodes()

    sp = SparkExecutor()

    sp.execute(spark_nodes)
