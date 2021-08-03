from pyspark.ml import Pipeline

from simple_repo.simple_spark.spark_node import Estimator


class SparkPipelineNode(Estimator):

    _stages = []

    def __init__(self, spark, lst, **kwargs):
        for k in lst:
            self._stages.append(k)
        super(SparkPipelineNode, self).__init__(spark, **kwargs)

    def execute(self):
        pipeline_stages = []
        for stage in self._stages:
            pipeline_stages.append(stage.execute())
        pipeline = Pipeline(stages=pipeline_stages)
        self.model = pipeline.fit(self.dataset)
