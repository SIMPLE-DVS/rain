from typing import List

from pyspark.ml import Pipeline

from simple_repo.nodes.spark.node_structure import Estimator, SparkNode


class SparkPipelineNode(Estimator):
    """Represent a Spark Pipeline consisting of SparkNode (stages)

    Parameters
    ----------
    stages: List[SparkNode]
        List of SparkNode that can be executed in a Spark Pipeline

    """

    _stages = []

    def __init__(self, node_id: str, stages: List[SparkNode]):
        super(SparkPipelineNode, self).__init__(node_id)
        for stage in stages:
            if stage.computational_instance is None:
                raise Exception(
                    "{} is not a valid stage".format(stage.__class__.__name__)
                )
            self._stages.append(stage)

    def execute(self):
        pipeline_stages = []
        for stage in self._stages:
            pipeline_stages.append(stage.computational_instance)
        pipeline = Pipeline(stages=pipeline_stages)
        self.model = pipeline.fit(self.dataset)
