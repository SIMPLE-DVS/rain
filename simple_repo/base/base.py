from abc import ABC, abstractmethod
import pandas as pd
import logger as lg


def get_step(step_id: str, step_list: list):
    """
    Utility function that given a step id and the list of steps, returns the step instance with the corresponding id.
    """
    # get all the nodes with the request id
    corr_steps = [step for step in step_list if step.step_id == step_id]

    if len(corr_steps) > 1:
        raise Exception(
            "Error! There are duplicated nodes with same id '{}'.".format(step_id)
        )
    elif len(corr_steps) < 1:
        raise Exception("Error! There aren't nodes with id '{}'.".format(step_id))

    return corr_steps[0]


class Node(object):
    """
    Class to represent, as a Python object, the configuration file.

        Parameters
        ----------

        idd : string
            The unique identifier that each node must have.

        name : string
            The full-name formed by \textit{package + module + class}, useful to dynamically import the
            module and to return the wanted class representing one step of the pipeline

        attr : dict
            List of features that characterizes each step of the pipeline. Obviously, depending on the node,
            we have a different structure of the list with different number of features.

        then : list
            List of idd representing the node(s) that are directly linked with the current node.

    """

    def __init__(self, idd: str, name: str, attr: dict, then: list):
        self._idd = idd
        self._name = name
        self._attr = attr
        self._then = then

    @property
    def id(self):
        return self._idd

    @property
    def name(self):
        return self._name

    @property
    def attr(self):
        return self._attr

    @property
    def then(self):
        return self._then


class PipelineStep(ABC):
    """
    Class implemented in order to represent the actual step of the pipeline that has to be performed.
    It is modeled as an abstract class that represents the root of the hierarchical structure of all the pipeline steps.
    """

    def __init__(self):
        self._step_id = None
        self._next_steps = []

    def add_step(self, step):
        self._next_steps.append(step)

        return self

    @abstractmethod
    def check_execution(self) -> bool:
        pass

    @abstractmethod
    def execute(self):
        pass

    @abstractmethod
    def communicate_result(self):
        pass

    @property
    def step_id(self):
        return self._step_id

    @step_id.setter
    def step_id(self, s_id: str):
        self._step_id = s_id

    @property
    def next_steps(self):
        return self._next_steps

    @next_steps.setter
    def next_steps(self, n_steps: str):
        self._next_steps = n_steps


class DataFrameManipulator(PipelineStep):
    """
    Class implemented in order to handle the manipulation of a DataFrame by taking one as input and returning its
    modified version as output.
    """

    def __init__(self):
        super(DataFrameManipulator, self).__init__()
        self._dataset = None

    def check_execution(self) -> bool:
        return self._dataset is not None

    @abstractmethod
    def execute(self):
        pass

    def communicate_result(self):
        for next_step in self.next_steps:
            if next_step.dataset is not None:
                lg.log_error(
                    "Cannot pass {}'s result to {}, it is used by another node.".format(
                        self.__class__.__name__, next_step.__class__.__name__
                    )
                )
                continue
            next_step.dataset = pd.DataFrame.copy(self._dataset)

        self._dataset = None

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, dset):
        self._dataset = dset


class ModelManipulator(PipelineStep):
    """
    Class implemented in order to handle the manipulation of a Machine Learning model by taking one as input and
    performing some specific tasks (e.g. pickle export).
    """

    def __init__(self):
        super(ModelManipulator, self).__init__()
        self._model = None

    def check_execution(self) -> bool:
        return self._model is not None

    @abstractmethod
    def execute(self):
        pass

    def communicate_result(self):
        for next_step in self.next_steps:
            if next_step.model is not None:
                lg.log_error(
                    "Cannot pass {}'s result to {}, it is used by another node.".format(
                        self.__class__.__name__, next_step.__class__.__name__
                    )
                )
                continue
            next_step.model = self._model

        self._model = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, mod):
        self._model = mod


class Trainer(PipelineStep):
    """
    Class implemented in order to handle the creation (and the consequent output) of a model by taking a DataFrame (
    training dataset) as input.
    """

    def __init__(self):
        super(Trainer, self).__init__()
        self._dataset = None
        self._model = None

    def check_execution(self) -> bool:
        return self._dataset is not None

    @abstractmethod
    def execute(self):
        pass

    def communicate_result(self):
        for next_step in self.next_steps:
            if next_step.model is not None:
                lg.log_error(
                    "Cannot pass {}'s result to {}, it is used by another node.".format(
                        self.__class__.__name__, next_step.__class__.__name__
                    )
                )
                continue
            next_step.model = self._model

        self._model = None
        self._dataset = None

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, dset):
        self._dataset = dset


class Predictor(PipelineStep):
    """
    Class implemented in order to handle the prediction over some data by taking a DataFrame (predict dataset) and a
    trained model as input, returning a DataFrame containing the predicted values.
    """

    def __init__(self):
        super(Predictor, self).__init__()
        self._dataset = None
        self._model = None

    def check_execution(self) -> bool:
        return self._dataset is not None and self._model is not None

    @abstractmethod
    def execute(self):
        pass

    def communicate_result(self):
        for next_step in self.next_steps:
            if next_step.dataset is not None:
                lg.log_error(
                    "Cannot pass {}'s result to {}, it is used by another node.".format(
                        self.__class__.__name__, next_step.__class__.__name__
                    )
                )
            next_step.dataset = pd.DataFrame.copy(self._dataset)

        self._dataset = None
        self._model = None

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, dset):
        self._dataset = dset

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, mod):
        self._model = mod
