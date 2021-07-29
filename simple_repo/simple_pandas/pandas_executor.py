import importlib
import json
from node_structure import PandasNode, PandasPipeline


def get_class(fullname: str) -> PandasNode:
    """
    Given a fullname formed by "package + module + class" (a.e. sigmalib.load.loader.CSVLoader)
    imports dynamically the module and returns the wanted <class>
    """

    full_name_parts = fullname.split(".")

    package_name = ".".join(full_name_parts[:-2])
    module_name = full_name_parts[-2]
    class_name = full_name_parts[-1]

    module = importlib.import_module("." + module_name, package_name)
    class_ = getattr(module, class_name)

    return class_


def load_config(config_file) -> dict:
    """
    Utility function that given a path, returns the json file representing the configuration of the pipeline.
    """
    with open(config_file, "r") as f:
        config = json.load(f)
        return config


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
