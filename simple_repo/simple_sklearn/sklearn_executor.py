import importlib
import json

import pandas
import sklearn.model_selection

from node_structure import SklearnEstimator


def get_class(fullname: str) -> SklearnEstimator:
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

    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    pd_config = load_config("pca_iris.json").get("sklearn")

    risultati = {}

    (
        risultati["d1_dt"],
        risultati["d2_dt"],
        risultati["d1_trg"],
        risultati["d2_trg"],
    ) = train_test_split(x, y, train_size=0.6)

    for node in pd_config:
        clazz = get_class(node.get("node"))

        inst = clazz(**node.get("parameters"))

        inst.fit_dataset = risultati.get("d1_dt")
        inst.fit_target = risultati.get("d1_trg")

        inst.fit()

        inst.score_dataset = risultati.get("d2_dt")
        inst.score_target = risultati.get("d2_trg")

        inst.score()

        print(inst.scores)
