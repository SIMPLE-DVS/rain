import pickle

from rain import OutputNode, InputNode
from rain.core.parameter import Parameters, KeyValueParameter


class PickleModelSave(OutputNode):
    """Node that stores a given object, for instance a trained model, in pickle format.

    Input
    -----
    model : pickle
        The object/model to store.

    Parameters
    ----------
    node_id : str
        Id of the node.
    path : str
        The path/filename where to store the object/model.
    """

    _input_vars = {"model": "pickle"}

    def __init__(self, node_id: str, path: str):
        super(PickleModelSave, self).__init__(node_id)

        self.parameters = Parameters(
            path=KeyValueParameter("path", str, path)
        )

    def execute(self):
        pickle.dump(self.model, open(self.parameters.path.value, "wb"))


class PickleModelLoad(InputNode):
    """Node that loads a given object, for instance a trained model, stored in pickle format.

    Output
    ------
    model : pickle
        The loaded object in pickle format.

    Parameters
    ----------
    node_id : str
        Id of the node.
    path : str
        The path of the stored object/model.
    """

    _output_vars = {"model": "pickle"}

    def __init__(self, node_id: str, path: str):
        super(PickleModelLoad, self).__init__(node_id)

        self.parameters = Parameters(
            path=KeyValueParameter("path", str, path),
        )

    def execute(self):
        self.model = pickle.load(open(self.parameters.path.value, "rb"))
