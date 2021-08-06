import json
from typing import List

from pyspark.sql import SparkSession

from simple_repo.simple_spark.spark_node import SparkNode, Transformer, Estimator
from simple_repo.base import get_class, reset


class SparkPipeline:

    def __init__(self, pipeline: List[SparkNode]):
        self._nodes = pipeline

    @property
    def nodes(self):
        return self._nodes

    def execute(self):
        if len(self._nodes) == 0:
            return None

        for i in range(0, len(self._nodes)):
            self._nodes[i].execute()

            if i + 1 > len(self._nodes) - 1:
                break

            try:
                if isinstance(self._nodes[i + 1], Transformer) or isinstance(self._nodes[i + 1], Estimator):
                    self._nodes[i + 1].dataset = self._nodes[i].dataset
                elif isinstance(self._nodes[i + 1], SparkNode):
                    self._nodes[i + 1].model = self._nodes[i].model
            except Exception as e:
                print(e)

            reset(self.nodes[i])


if __name__ == '__main__':
    spark = SparkSession.builder.getOrCreate()
    with open("./spark_conf.json", "r") as f:
        config = json.load(f)

    spark_nodes = config.get("sparkNode")
    sp = get_class("pipeline.spark_pipeline.SparkPipelineNode")
    nodes = []

    for n in spark_nodes:
        cls = get_class(n.get("name"))
        if cls == sp:
            stages = []
            pipe = n.get("attr").get("stages")
            for s in pipe:
                c = get_class(s.get("name"))
                stage = c(spark=spark, **s.get("param"))
                stages.append(stage)
            node = cls(spark=spark, lst=stages)
            nodes.append(node)
        else:
            node = cls(spark=spark, **n.get("attr"))
            nodes.append(node)

    p = SparkPipeline(nodes)
    p.execute()
