from node_structure import PandasPipeline
from simple_repo.base import get_class, load_config


if __name__ == "__main__":
    pd_config = load_config("pandas_config.json").get("pandas")

    stages = []

    for node in pd_config.get("stages"):
        node_class = get_class(node.get("node"))

        node_inst = node_class(**node.get("parameters"))

        stages.append(node_inst)

    pandas_pipeline = PandasPipeline(stages)

    print(pd_config)
    print(pandas_pipeline.execute())
