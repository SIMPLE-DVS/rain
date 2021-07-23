from simple_repo.base import DataFrameManipulator
from typing import List
from sklearn.model_selection import train_test_split
from simple_repo.base import get_step
import pandas as pd
import simple_repo.base.logger as lg

__all__ = ["Filter", "Pivot", "SplitTrainTest"]


def _input_string_list(val):
    if isinstance(val, list):
        return val
    elif isinstance(val, str):
        return [val]
    else:
        return None


def _filter_column_by_value(dataset, column: str, value):
    """
    Utility function used, given a dataset, to filter a specific column with a specific value
    """
    dataset = dataset[dataset[column] == value]
    return dataset


def _set_column_type(dataset, column: str, column_type: str):
    """
    Utility function used, given a dataset, to set the type of a specific column
    """
    if column_type == "string":
        dataset[column] = dataset[column].astype(column_type)
    elif column_type == "timedelta":
        dataset[column] = pd.to_timedelta(dataset[column])
        dataset[column] = dataset[column].apply(lambda elem: elem.total_seconds())
    elif column_type == "datetime":
        dataset[column] = pd.to_datetime(dataset[column]).dt.date
    return dataset


def _filter_feature(dataset, feature):
    """
    Utility function used to manipulate a dataset and apply the given feature
    """
    try:
        dataset = _set_column_type(dataset, feature["code"], feature["type"])
    except Exception:
        pass
    try:
        dataset = _filter_column_by_value(
            dataset, feature["code"], feature["filter_value"]
        )
    except Exception:
        pass
    return dataset


class Filter(DataFrameManipulator):
    """
    Class that represents a step of the pipeline that, given a dataset, performs the request data preprocess.

        Parameters
        ----------

        **attr : dict
            All the features used to manipulate and preprocess the dataset.
    """

    def __init__(self, **attr: dict):
        super(Filter, self).__init__()
        lg.log_info(self, "Creating a Filter.")
        self._attr = attr
        lg.log_info_param(self, **self._attr)

    def execute(self):
        lg.log_info(self, "Executing the Filter step with id {}.".format(self.step_id))
        self._dataset = self.filter()

    def filter(self):
        """
        Method used to apply the feature preprocessing to the given dataset.
        """
        lg.log_debug(
            self, "\nInput dataset (head):\n{}\n".format(self._dataset.head(5))
        )
        features = self._attr["features"]
        if "*" in [x["code"] for x in features]:
            for feature in features:
                self._dataset = _filter_feature(self._dataset, feature)
        else:
            features_list = []
            for feature in features:
                self._dataset = _filter_feature(self._dataset, feature)
                features_list.append(feature["code"])
            self._dataset = self._dataset.filter(features_list)
        lg.log_debug(self, "Columns tpye:\n{}\n".format(self._dataset.dtypes))
        lg.log_debug(
            self, "Filtered dataset (head):\n{}\n".format(self._dataset.head(5))
        )
        return self._dataset


class Pivot(DataFrameManipulator):
    """
    Class that represents a step of the pipeline that, given a dataset, returns a pivot table

        Parameters
        ----------

        **param : dict
            All the optional parameters that can be passed to the Pandas pivot_table method. They can be found at
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.pivot_table.html
    """

    def __init__(self, **param: dict):
        super(Pivot, self).__init__()
        lg.log_info(self, "Creating a Pivot.")
        self._param = param
        lg.log_info_param(self, **self._param)

    def execute(self):
        lg.log_info(self, "Executing the Pivot step with id {}.".format(self.step_id))
        self._dataset = pd.pivot_table(self._dataset, **self._param)
        # self._dataset, values=self._data, index=self._row, columns=self._column, aggfunc=self._agg, fill_value=0


class SplitTrainTest(DataFrameManipulator):
    """
    Class that represents a step of the pipeline used to split a dataframe into train and test datasets.

        Parameters
        ----------

        **param : dict
            All the optional parameters that can be passed to the sklearn train_test_split. They can be found at
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """

    def __init__(
        self, send_train_to: List[str], send_test_to: List[str], **param: dict
    ):
        super(SplitTrainTest, self).__init__()
        lg.log_info(self, "Creating a SplitTrainTest.")

        self._splitted_train_dataset = None
        self._splitted_test_dataset = None
        self._send_train_to = _input_string_list(send_train_to)
        self._send_test_to = _input_string_list(send_test_to)
        self._split_attr = param

        lg.log_info_param(
            self,
            send_train_to=self._send_train_to,
            send_test_to=self._send_test_to,
            **self._split_attr
        )

    def communicate_result(self):
        for train_rec_id in self._send_train_to:
            train_receiver = get_step(train_rec_id, self._next_steps)
            if train_receiver.dataset is not None:
                lg.log_error(
                    "Cannot pass {}'s result to {}, it is used by another node.".format(
                        self.__class__.__name__, train_receiver.__class__.__name__
                    ),
                    self,
                )
                continue
            train_receiver.dataset = pd.DataFrame.copy(self._splitted_train_dataset)

        for test_rec_id in self._send_test_to:
            test_receiver = get_step(test_rec_id, self._next_steps)
            if test_receiver.dataset is not None:
                lg.log_error(
                    "Cannot pass {}'s result to {}, it is used by another node.".format(
                        self.__class__.__name__, test_receiver.__class__.__name__
                    ),
                    self,
                )
                continue
            test_receiver.dataset = pd.DataFrame.copy(self._splitted_test_dataset)

        self._dataset = None

    def execute(self):
        lg.log_info(
            self, "Executing the SplitTrainTests step with id {}.".format(self.step_id)
        )
        self._splitted_train_dataset, self._splitted_test_dataset = train_test_split(
            self._dataset, **self._split_attr
        )
