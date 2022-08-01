from abc import abstractmethod
from typing import Union

import pandas
import pandas as pd

from rain.core.base import InputNode, OutputNode, Tags, LibTag, TypeTag
from rain.core.parameter import KeyValueParameter, Parameters


class PandasInputNode(InputNode):
    """Parent class for all the nodes that load a pandas DataFrame from some kind of source.
    """
    _output_vars = {"dataset": pandas.DataFrame}

    @abstractmethod
    def execute(self):
        pass  # pragma: no cover

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.PANDAS, TypeTag.INPUT)


class PandasOutputNode(OutputNode):
    """Parent class for all the nodes that return a pandas DataFrame toward some kind of destination.
    """
    _input_vars = {"dataset": pandas.DataFrame}

    @abstractmethod
    def execute(self):
        pass  # pragma: no cover

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.PANDAS, TypeTag.OUTPUT)


class PandasCSVLoader(PandasInputNode):
    """Loads a pandas DataFrame from a CSV file.

    Output
    ------
    dataset : pandas.DataFrame
        The loaded csv file as a pandas DataFrame.

    Parameters
    ----------
    path : str
        Of the CSV file.
    delim : str, default ','
        Delimiter symbol of the CSV file.
    index_col: Union[int, str], default=None
        Column(s) to use as the row labels of the DataFrame, either given as string name or column index.

    Notes
    -----
    Visit `<https://pandas.pydata.org/pandas-docs/version/1.3/reference/api/pandas.read_csv.html>`_ for Pandas read_csv
    documentation.
    """

    # _parameters = { "filepath_or_buffer": PandasParameter("filepath_or_buffer", str, is_mandatory=True),
    # sep=<no_default>, delimiter=None, header='infer', names=<no_default>, index_col=None, usecols=None,
    # squeeze=False, prefix=<no_default>, mangle_dupe_cols=True, dtype=None, engine=None, converters=None,
    # true_values=None, false_values=None, skipinitialspace=False, skiprows=None, skipfooter=0, nrows=None,
    # na_values=None, keep_default_na=True, na_filter=True, verbose=False, skip_blank_lines=True, parse_dates=False,
    # infer_datetime_format=False, keep_date_col=False, date_parser=None, dayfirst=False, cache_dates=True,
    # iterator=False, chunksize=None, compression='infer', thousands=None, decimal='.', lineterminator=None,
    # quotechar='"', quoting=0, doublequote=True, escapechar=None, comment=None, encoding=None,
    # encoding_errors='strict', dialect=None, error_bad_lines=None, warn_bad_lines=None, on_bad_lines=None,
    # delim_whitespace=False, low_memory=True, memory_map=False, float_precision=None, storage_options=None }

    def __init__(self, node_id: str, path: str, delim: str = ",", index_col: Union[int, str] = None):
        super(PandasCSVLoader, self).__init__(node_id)

        self.parameters = Parameters(
            path=KeyValueParameter("filepath_or_buffer", str, path),
            delim=KeyValueParameter("delimiter", str, delim),
            index_col=KeyValueParameter("index_col", str, index_col)
        )

        self.parameters.group_all("read_csv")

    def execute(self):
        param_dict = self.parameters.get_dict_from_group("read_csv")
        self.dataset = pandas.read_csv(**param_dict)


class PandasCSVWriter(PandasOutputNode):
    """Writes a pandas DataFrame into a CSV file.

    Input
    -----
    dataset : pandas.DataFrame
        The pandas DataFrame to write in a CSV file.

    Parameters
    ----------
    path : str
        Of the CSV file.
    delim : str, default ','
        Delimiter symbol of the CSV file.
    include_rows : bool, default True
        Whether to include rows indexes.
    rows_column_label : str, default None
        If rows indexes must be included you can give a name to its column.
    include_columns : bool, default True
        Whether to include column names.
    columns : list[str], default None
        If column names must be included you can give names to them.
        The order is relevant.

    Notes
    -----
    Visit `<https://pandas.pydata.org/pandas-docs/version/1.3/reference/api/pandas.DataFrame.to_csv.html>`_ for Pandas
    to_csv documentation.
    """

    # _parameters = { "filepath_or_buffer": PandasParameter("filepath_or_buffer", str, is_mandatory=True),
    # sep=<no_default>, delimiter=None, header='infer', names=<no_default>, index_col=None, usecols=None,
    # squeeze=False, prefix=<no_default>, mangle_dupe_cols=True, dtype=None, engine=None, converters=None,
    # true_values=None, false_values=None, skipinitialspace=False, skiprows=None, skipfooter=0, nrows=None,
    # na_values=None, keep_default_na=True, na_filter=True, verbose=False, skip_blank_lines=True, parse_dates=False,
    # infer_datetime_format=False, keep_date_col=False, date_parser=None, dayfirst=False, cache_dates=True,
    # iterator=False, chunksize=None, compression='infer', thousands=None, decimal='.', lineterminator=None,
    # quotechar='"', quoting=0, doublequote=True, escapechar=None, comment=None, encoding=None,
    # encoding_errors='strict', dialect=None, error_bad_lines=None, warn_bad_lines=None, on_bad_lines=None,
    # delim_whitespace=False, low_memory=True, memory_map=False, float_precision=None, storage_options=None }

    def __init__(
        self,
        node_id: str,
        path: str,
        delim: str = ",",
        include_rows: bool = True,
        rows_column_label: str = None,
        include_columns: bool = True,
        columns: list = None,
    ):
        super(PandasCSVWriter, self).__init__(node_id)
        self.parameters = Parameters(
            path=KeyValueParameter("path_or_buf", str, path),
            delim=KeyValueParameter("sep", str, delim),
            include_rows=KeyValueParameter("index", bool, include_rows),
            rows_column_label=KeyValueParameter("index_label", str, rows_column_label),
            include_columns=KeyValueParameter("header", bool, include_columns),
            columns=KeyValueParameter("columns", list, columns),
        )

        self.parameters.group_all("write_csv")

    def execute(self):
        param_dict = self.parameters.get_dict_from_group("write_csv")

        if not isinstance(self.dataset, pd.DataFrame):
            self.dataset = pd.DataFrame(self.dataset)

        self.dataset.to_csv(**param_dict)
