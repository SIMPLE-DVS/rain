from typing import List

from pyspark.ml import Pipeline

from simple_repo.simple_spark.node_structure import Estimator, SparkNode


class SparkPipelineNode(Estimator):
    """Represent a Spark Pipeline consisting of SparkNode (stages)

    Parameters
    ----------
    stages: list of SparkNode
        List of SparkNode that can be executed in a Spark Pipeline

    """

    _stages = []

    def __init__(self, stages: List[SparkNode]):
        for stage in stages:
            if stage.computational_instance is None:
                raise Exception("{} is not a valid stage".format(stage.__class__.__name__))
            self._stages.append(stage)
        super(SparkPipelineNode, self).__init__()

    def execute(self):
        pipeline_stages = []
        for stage in self._stages:
            pipeline_stages.append(stage.computational_instance)
        pipeline = Pipeline(stages=pipeline_stages)
        self.model = pipeline.fit(self.dataset)
