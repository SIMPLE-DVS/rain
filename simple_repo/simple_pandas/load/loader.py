from simple_repo.commons import DataFrameManipulator, ModelManipulator
import pandas as pd
import pickle
import simple_repo.logger as lg


class CSVLoader(DataFrameManipulator):
    """
    Class that represents a step of the pipeline that loads a dataset stored in .csv format

        Parameters
        ----------

        path : str
            Path where the dataset is stored.

        param : dict
            All the optional parameters that can be passed to the Pandas read_csv method.
            They can be found at https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    """

    def __init__(self, path: str, **param: dict):
        super(CSVLoader, self).__init__()
        lg.log_info(self, "Creating a CSVLoader.")

        self._path = path
        self._csvloader_attr = param

        lg.log_info_param(self, path=self._path, **self._csvloader_attr)

    def check_execution(self) -> bool:
        return self._path is not None

    def execute(self):
        self._dataset = pd.read_csv(self._path, **self._csvloader_attr)
        lg.log_debug(self, "Input dataset (head):\n{}\n".format(self._dataset.head(5)))


class JSONLoader(DataFrameManipulator):
    """
    Class that represents a step of the pipeline that loads a dataset stored in .json format

        Parameters
        ----------

        path : str
             Path where the dataset is stored.

        param : dict
            All the optional parameters that can be passed to the Pandas read_json method.
            They can be found at https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_json.html
    """

    def __init__(self, path: str, **param: dict):
        super(JSONLoader, self).__init__()
        lg.log_info(self, "Creating an JSONLoader.")

        self._path = path
        self._jsonloader_attr = param

        lg.log_info_param(self, path=self._path, **self._jsonloader_attr)

    def check_execution(self) -> bool:
        return self._path is not None

    def execute(self):
        self._dataset = pd.read_json(self._path, **self._jsonloader_attr)
        lg.log_debug(self, "Input dataset (head):\n{}\n".format(self._dataset.head(5)))


class PickleLoader(ModelManipulator):
    """
    Class that represents a step of the pipeline that loads a model stored in .pkl format

        Parameters
        ----------

        path : str
             Path where the model is stored.

        param : dict
            All the optional parameters that can be passed to the pickle load method.
            They can be found at https://docs.python.org/3/library/pickle.html#pickle.load
    """

    def __init__(self, path: str, **param: dict):
        super(PickleLoader, self).__init__()
        lg.log_info(self, "Creating an PickleLoader.")

        self._path = path
        self._pickleload_attr = param

        lg.log_info_param(self, path=self._path, **self._pickleload_attr)

    def check_execution(self) -> bool:
        return self._path is not None

    def execute(self):
        self._model = pickle.load(open(self._path, "rb"), **self._pickleload_attr)
        lg.log_debug(self, "Model {} loaded\n".format(self.model))
