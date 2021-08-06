import os
import pickle as pkl
from datetime import datetime
from simple_repo.commons import DataFrameManipulator, ModelManipulator
import simple_repo.logger as lg


def check_dir(path):
    """
    Utility method to create the directory if it does not exist
    """
    full_path = path.split("/") if "/" in path else path.split("\\")
    dir_path = "/".join(full_path[:-1])
    if not os.path.isfile(dir_path) and not os.path.exists(dir_path):
        os.makedirs(dir_path)


class OutDataframe(DataFrameManipulator):
    """
    Class that represents a step of the pipeline that stores, in the specific path, the given dataset in a .csv format.

        Parameters
        ----------

        path : string
            Path where to store the dataset in .csv format.

        param : dict
            All the optional parameters that can be passed to the Pandas to_csv method. They can be found at
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html
    """

    def __init__(self, path: str, add_date: bool = False, **param: dict):
        super(OutDataframe, self).__init__()
        lg.log_info(self, "Creating an OutDataframe.")
        self._path = path
        self._add_date = add_date
        self._param = param
        lg.log_info_param(self, path=path, **self._param)

    def execute(self) -> any:
        check_dir(self._path)
        date = datetime.now().strftime("_%d-%m-%Y_%H-%M-%S") if self._add_date else ""
        self._dataset.to_csv("{}{}.csv".format(self._path, date), **self._param)
        lg.log_info(self, "Dataset stored in {}{}.csv".format(self._path, date))


class OutModel(ModelManipulator):
    """
    Class that represents a step of the pipeline that stores, in the specific path, the given ML model in a .pkl format.

        Parameters
        ----------

        path : string
            Path where to store the model in .pkl format.

        param : dict
            All the optional parameters that can be passed to the Pickle dump method. They can be found at
            https://docs.python.org/3/library/pickle.html#pickle.dump

    """

    def __init__(self, path, add_date: bool = False, **param: dict):
        super(OutModel, self).__init__()
        lg.log_info(self, "Creating an OutModel.")
        self._path = path
        self._add_date = add_date
        self._param = param
        lg.log_info_param(self, path=path, **self._param)

    def execute(self) -> any:
        lg.log_info(self, "Executing the OutModel with id {}.".format(self.step_id))
        check_dir(self._path)
        date = datetime.now().strftime("_%d-%m-%Y_%H-%M-%S") if self._add_date else ""
        pkl.dump(
            self._model, open("{}{}.pkl".format(self._path, date), "wb"), **self._param
        )
        lg.log_info(self, "Model stored in {}{}.pkl".format(self._path, date))
