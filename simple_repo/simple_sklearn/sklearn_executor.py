from simple_repo.base import get_class, load_config
from sklearn import datasets
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

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

        inst = clazz(node.get("execute"), **node.get("parameters"))

        inst.fit_dataset = risultati.get("d1_dt")
        inst.fit_target = risultati.get("d1_trg")

        inst.score_dataset = risultati.get("d2_dt")
        inst.score_target = risultati.get("d2_trg")

        inst.execute()

        print(inst.scores)
