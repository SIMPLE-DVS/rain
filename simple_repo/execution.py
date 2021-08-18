from collections import OrderedDict
from typing import List, Tuple

from simple_repo.base import Singleton, Node
from simple_repo.dag import SimpleJSONParser
from simple_repo.simple_pandas.node_structure import PandasExecutor
from simple_repo.simple_sklearn.node_structure import SklearnExecutor
from simple_repo.simple_spark.spark_manager import SparkExecutor


def reset(simple_node):
    dic = vars(simple_node)
    for i in dic.keys():
        dic[i] = None


class SimpleSubPipeline:
    def __init__(self, s_type, nodes: List[Node], executor):
        self._type = s_type
        self._nodes = OrderedDict()
        self._nexts = {}
        self._foreign_nexts = {}
        self._executor = executor

        s_nodes, nxts = self._executor.convert(nodes)

        self._nodes.update(s_nodes)

        if nxts is not None:
            self._nexts.update(nxts)

    def execute_subpipeline(self):
        for node_name, node in self._nodes.items():
            self._executor.execute(node)

            if node_name not in self._nexts.keys():
                continue

            f_nexts = []

            for nxt in self._nexts.get(node_name):
                receiver_id = nxt.get("send_to")

                receiver = self._nodes.get(receiver_id)

                if receiver is None:
                    f_nexts.append(nxt)
                    continue

                for k, v in nxt.items():
                    if k == "send_to":
                        continue

                    actual_node_output = node.get_output_value(k)
                    receiver.set_input_value(v, actual_node_output)

            if f_nexts:
                self._foreign_nexts[node_name] = f_nexts

    def reset(self):
        for s_node in self._nodes.values():
            reset(s_node)

    @property
    def executor(self):
        return self._executor

    @property
    def nodes(self):
        return self._nodes

    @property
    def foreign_nexts(self):
        return self._foreign_nexts


class SimplePipeline:
    def __init__(self):
        self._subpipelines: List[SimpleSubPipeline] = []
        self._nodes = OrderedDict()

    def add_subpipeline(self, subpipeline: SimpleSubPipeline):
        self._subpipelines.append(subpipeline)
        self._nodes.update(subpipeline.nodes)

    def execute(self):
        for subpip in self._subpipelines:
            subpip.execute_subpipeline()

            if not subpip.foreign_nexts:
                continue

            for node_name, nxts in subpip.foreign_nexts.items():
                node = self._nodes.get(node_name)
                for nxt in nxts:
                    receiver_id = nxt.get("send_to")

                    receiver = self._nodes.get(receiver_id)

                    for k, v in nxt.items():
                        if k == "send_to":
                            continue

                        actual_node_output = node.get_output_value(k)
                        receiver.set_input_value(v, actual_node_output)

            subpip.reset()

    @property
    def subpipelines(self):
        return self._subpipelines


if __name__ == "__main__":
    sjp = SimpleJSONParser()

    sjp.parse_configuration("simple_spark/spark_conf.json")

    # sjp.show_dag()

    pipeline = sjp.get_subpipelines()

    executors = {
        "pandas": PandasExecutor(),
        "sklearn": SklearnExecutor(),
        "spark": SparkExecutor(),
    }

    pippo = SimplePipeline()

    for subpip_type, subpip_node_list in pipeline:
        pippo.add_subpipeline(
            SimpleSubPipeline(subpip_type, subpip_node_list, executors.get(subpip_type))
        )

    pippo.execute()
