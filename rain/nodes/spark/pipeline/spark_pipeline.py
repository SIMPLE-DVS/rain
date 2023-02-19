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

from typing import List

from pyspark.ml import Pipeline

from rain.nodes.spark.node_structure import Estimator, SparkNode


class SparkPipelineNode(Estimator):
    """Represent a Spark Pipeline consisting of SparkNode (stages). It should contain some Spark Transformer and a final
    Spark Estimator that return the trained model.

    Input
    -----
    dataset : DataFrame
        A Spark DataFrame.

    Output
    ------
    model : PipelineModel
        A Spark PipelineModel.

    Parameters
    ----------
    node_id : str
        Id of the node.
    stages: List[SparkNode]
        List of SparkNode that can be executed in a Spark Pipeline.

    Notes
    -----
    Visit `<https://spark.apache.org/docs/latest/ml-pipeline.html#pipeline>`_ for Spark Pipeline documentation.

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
