from typing import List

from pyspark.ml import Pipeline

from simple_repo.simple_spark.node_structure import Estimator, SparkNodeSession


class SparkPipelineNode(Estimator):
    """Represent a Spark Pipeline consisting of SparkNode (stages)

    Parameters
    ----------
    stages: str
        List of SparkNode that can be executed in a Spark Pipeline

    """

    _stages = []

    def __init__(self, stages: List[SparkNodeSession]):
        for stage in stages:
            self._stages.append(stage)
        super(SparkPipelineNode, self).__init__()

    def execute(self):
        pipeline_stages = []
        for stage in self._stages:
            pipeline_stages.append(stage.execute())
        pipeline = Pipeline(stages=pipeline_stages)
        self.model = pipeline.fit(self.dataset)
