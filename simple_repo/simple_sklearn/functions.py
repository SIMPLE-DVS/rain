from simple_repo.simple_sklearn.node_structure import SklearnFunction


class SimpleSplitTrainTest(SklearnFunction):
    def __init__(self, node_id: str, **kwargs):
        super(SimpleSplitTrainTest, self).__init__(**kwargs)
