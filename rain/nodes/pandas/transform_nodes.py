from typing import List, Tuple, Any

import numpy
import pandas
import pandas as pd

from rain.core.exception import ParametersException, PandasSequenceException
from rain.core.parameter import KeyValueParameter, Parameters
from rain.nodes.pandas.node_structure import PandasTransformer, PandasNode


# def _filter_column_by_value(dataset, column: str, value):
#     """
#     Utility function used, given a dataset, to filter a specific column with a specific value
#     """
#     dataset = dataset[dataset[column] == value]
#     return dataset
#
#
# def _set_column_type(dataset, column: str, column_type: str):
#     """
#     Utility function used, given a dataset, to set the type of a specific column
#     """
#     if column_type == "string":
#         dataset[column] = dataset[column].astype(column_type)
#     elif column_type == "timedelta":
#         dataset[column] = pandas.to_timedelta(dataset[column])
#         dataset[column] = dataset[column].apply(lambda elem: elem.total_seconds())
#     elif column_type == "datetime":
#         dataset[column] = pandas.to_datetime(dataset[column]).dt.date
#     return dataset
#
#
# def _filter_feature(dataset, feature):
#     """
#     Utility function used to manipulate a dataset and apply the given feature
#     """
#     try:
#         dataset = _set_column_type(dataset, feature["name"], feature["type"])
#     except Exception:
#         pass
#     try:
#         dataset = _filter_column_by_value(
#             dataset, feature["name"], feature["filter_value"]
#         )
#     except Exception:
#         pass
#     return dataset
#
#
# def _filter(self):
#     """
#     Method used to apply the feature preprocessing to the given dataset.
#     """
#     features = self.parameters.columns.parameters
#     if "*" in [x["name"] for x in features]:
#         for feature in features:
#             self.dataset = _filter_feature(self.dataset, feature)
#     else:
#         features_list = []
#         for feature in features:
#             self.dataset = _filter_feature(self.dataset, feature)
#             features_list.append(feature["name"])
#         self.dataset = self.dataset.filter(features_list)
#     return self.dataset


class PandasColumnsFiltering(PandasTransformer):
    """PandasColumnsFiltering manages filtering of columns. This node gives access
    to several functionalities such as:
    - select columns by their indexes;
    - select columns by their names (labels);
    - select columns containing a substring in their names;
    - select columns that match a regex;
    - select columns in a range of indexes;
    - assign a type to a column.
    Every parameter but 'columns_type' is mutually exclusive, meaning that only one can be used.

    Parameters
    ----------
    node_id : str
        Id of the node.
    column_indexes : List[int]
        Filters the dataset selecting the given indexes. Uses the pandas iloc function.
    column_names : List[str]
        Filters the dataset selecting the given column labels. Uses the pandas filter function.
    columns_like : str
        Keep columns for which the given string is a substring of the column label.
    columns_regex : str
        Keep columns for which column labels match a given pattern.
    columns_range : Tuple[int, int]
        Keep columns for which index falls withing the given range (from, to (excluded)).
    columns_type : str or List[str]
        Type to assign to columns. It can be either a string, meaning that it will try to apply
        the chosen type to all the columns, or a list of strings, one for each column,
        meaning that it will try to assign a chosen type to each column in order.
    """

    def __init__(
        self,
        node_id: str,
        column_indexes: List[int] = None,
        column_names: List[str] = None,
        columns_like: str = None,
        columns_regex: str = None,
        columns_range: Tuple[int, int] = None,
        columns_type=None,
    ):
        super(PandasColumnsFiltering, self).__init__(node_id)

        self.none_parameters_count = sum(
            parameter is not None
            for parameter in [
                column_indexes,
                column_names,
                columns_like,
                columns_regex,
                columns_range,
            ]
        )

        if self.none_parameters_count > 1:
            raise ParametersException("Filtering parameters are mutually exclusive.")

        self.parameters = Parameters(
            columns_range=KeyValueParameter("range", str, value=columns_range),
            column_indexes=KeyValueParameter("indexes", str, value=column_indexes),
            column_names=KeyValueParameter("items", str, value=column_names),
            columns_like=KeyValueParameter("like", str, value=columns_like),
            axis=KeyValueParameter("axis", str, value="columns"),
            columns_regex=KeyValueParameter("regex", str, value=columns_regex),
            columns_type=KeyValueParameter("ctype", list, value=columns_type),
        )

        self.parameters.add_group(
            "filter", keys=["column_names", "columns_like", "columns_regex", "axis"]
        )

    def execute(self):
        if self.parameters.column_indexes.value:
            self.dataset = self.dataset.iloc[:, self.parameters.column_indexes.value]
        elif self.parameters.columns_range.value:
            from_var = self.parameters.columns_range.value[0]
            to_var = self.parameters.columns_range.value[1]
            self.dataset = self.dataset.iloc[:, from_var:to_var]
        elif self.none_parameters_count == 1:
            self.dataset = self.dataset.filter(
                **self.parameters.get_dict_from_group("filter")
            )

        if (col_type := self.parameters.columns_type.value) is not None:
            if isinstance(col_type, str):
                self.dataset.astype(col_type)
            elif isinstance(col_type, list):
                self.dataset = self.dataset.astype(
                    {
                        col: col_type[index]
                        for index, col in enumerate(self.dataset.columns)
                        if col_type[index] is not None
                    }
                )


class PandasSelectRows(PandasNode):
    """PandasSelectRows manages selection of rows, which can later be filtered or deleted.

    Parameters
    ----------
    node_id : str
        Id of the node.
    select_nan : bool, default False
        Select rows with at least one NaN value.
    conditions : List[str]
        List of conditions to select rows.
    """

    _input_vars = {"dataset": pandas.DataFrame}
    _output_vars = {"selection": pandas.Series}

    def __init__(
        self,
        node_id: str,
        select_nan: bool = False,
        conditions: List[str] = None,
    ):
        super(PandasSelectRows, self).__init__(node_id)

        self.parameters = Parameters(
            select_nan=KeyValueParameter("select_nan", str, value=select_nan),
            conditions=KeyValueParameter("conditions", str, value=conditions),
        )

    def execute(self):
        if self.parameters.select_nan.value:
            self.selection = self.dataset.isnull().any(axis=1)
        if conditions := self.parameters.conditions.value:
            conds = []
            for cond in conditions:
                conds_or = [splitted_cond.strip() for splitted_cond in cond.split("&")]
                new_cond = ["self.dataset.{}".format(or_cond) for or_cond in conds_or]
                new_cond = "({})".format(" & ".join(new_cond))
                conds.append(new_cond)
            conds = " | ".join(conds)
            self.selection = pandas.eval(conds, target=self.dataset)


class PandasFilterRows(PandasTransformer):
    """PandasFilterRows manages filtering of rows that have been previously selected.

    Parameters
    ----------
    node_id : str
        Id of the node.
    """

    _input_vars = {"selected_rows": pandas.Series}

    def __init__(
        self,
        node_id: str,
    ):
        super(PandasFilterRows, self).__init__(node_id)

    def execute(self):
        self.dataset = self.dataset[self.selected_rows]


class PandasDropNan(PandasTransformer):
    """Drops rows or columns that either only contains a nan or that has all nan values.

    Parameters
    ----------
    node_id : str
        Id of the node.
    axis : {'rows', 'columns'}, default 'rows'
        The axis from where to remove the nan values.
    how : {'any', 'all'}, default 'any'
        Whether to remove a row or a column which either contains any nan value or
        contains all nan values.
    """

    def __init__(
        self,
        node_id: str,
        axis="rows",
        how="any",
    ):
        super(PandasDropNan, self).__init__(node_id)
        if not axis == "rows" and not axis == "columns":
            raise AttributeError("Invalid value for 'axis', set 'rows' or 'columns'.")

        axis = 0 if axis == "rows" else 1

        self.parameters = Parameters(
            axis=KeyValueParameter("axis", str, axis),
            how=KeyValueParameter("how", str, how),
        )

    def execute(self):
        self.dataset = self.dataset.dropna(**self.parameters.get_dict())


class PandasPivot(PandasTransformer):
    """Transforms a DataFrame into a Pivot from the given rows, columns and values.

    Parameters
    ----------
    rows : str
        Name of the column whose values will be the rows of the pivot.
    columns : str
        Name of the column whose values will be the columns of the pivot.
    values : str
        Name of the column whose values will be the values of the pivot.
    aggfunc : str, default 'mean'
        Function to use for the aggregation.
    fill_value : int, default 0
        Value to replace missing values with.
    dropna : bool, default True
        Do not include columns whose entries are all NaN.
    sort : bool, default True
        Specifies if the result should be sorted.
    """

    def __init__(
        self,
        node_id: str,
        rows: str,
        columns: str,
        values: str,
        aggfunc: str = "mean",
        fill_value: int = 0,
        dropna: bool = True,
        sort: bool = True,
    ):
        super(PandasPivot, self).__init__(node_id)
        self.parameters = Parameters(
            rows=KeyValueParameter("index", str, rows),
            columns=KeyValueParameter("columns", str, columns),
            values=KeyValueParameter("values", str, values),
            aggfunc=KeyValueParameter("aggfunc", str, aggfunc),
            fill_value=KeyValueParameter("fill_value", int, fill_value),
            dropna=KeyValueParameter("dropna", bool, dropna),
            sort=KeyValueParameter("sort", bool, sort),
        )

    def execute(self):
        param_dict = self.parameters.get_dict()
        self.dataset = pandas.pivot_table(self.dataset, **param_dict)


class PandasRenameColumn(PandasTransformer):
    """Sets column names for a pandas DataFrame.

    Parameters
    ----------
    columns : list[str]
        Column names to assign to the DataFrame.
        The order is relevant.
    """

    def __init__(self, node_id: str, columns: list):
        super(PandasRenameColumn, self).__init__(node_id)
        self.parameters = Parameters(columns=KeyValueParameter("col", list, columns))

    def execute(self):
        cols = self.parameters.columns.value

        if not isinstance(self.dataset, pandas.DataFrame):
            self.dataset = pandas.DataFrame(self.dataset)

        self.dataset.columns = cols


class PandasSequence(PandasTransformer):
    """
    PandasSequence wraps a list of nodes that must be executed in sequence into a single node.
    Intermediate values are passed along the chain using the 'dataset' variable, hence only
    PandasNodes can be used within a sequence.

    Parameters
    ----------
    node_id : str
        The unique id of the node.
    stages : list of PandasTransformer
        ordered in an execution sequence. They must all be PandasNodes, hence have a 'dataset'
        variable used for input and output.
    """

    def __init__(self, node_id: str, stages: List[PandasTransformer]):
        super(PandasSequence, self).__init__(node_id)

        if not all(isinstance(stage, PandasTransformer) for stage in stages):
            raise PandasSequenceException("Every stage must be a PandasNode.")

        self._stages = stages

    def execute(self):
        for stage in self._stages:
            stage.set_input_value("dataset", self.dataset)
            stage.execute()
            self.dataset = stage.get_output_value("dataset")


class PandasAddColumn(PandasTransformer):
    """
    Node used to add a column to a Pandas Dataframe starting from a given Pandas Series.

    Parameters
    ----------
    node_id : str
        The unique id of the node.
    loc : int
        Insertion index. Must verify 0 <= loc <= len(columns)
    col : str
        Label of the inserted column.
    """

    _input_vars = {"column": pd.Series}

    def __init__(self, node_id: str, loc: int, col: str):
        super(PandasAddColumn, self).__init__(node_id)
        self.parameters = Parameters(
            loc=KeyValueParameter("loc", int, loc),
            col=KeyValueParameter("column", str, col),
        )

    def execute(self):
        if self.parameters.loc.value > len(self.dataset.columns):
            self.parameters.loc.value = len(self.dataset.columns)
        self.dataset.insert(value=self.column, **self.parameters.get_dict())


class PandasReplaceColumn(PandasNode):
    """
    Node used to replace the boolean values of a Pandas Series with other values given by the user.

    Parameters
    ----------
    node_id : str
        The unique id of the node.
    first_value : Any
        Value used when the condition is True.
    second_value : Any
        Value used when the condition is True.
    """

    _input_vars = {"column": pd.Series}
    _output_vars = {"column": pd.Series}

    def __init__(self, node_id: str, first_value: Any, second_value: Any):
        super(PandasReplaceColumn, self).__init__(node_id)
        self.parameters = Parameters(
            first_value=KeyValueParameter("first_value", Any, first_value),
            second_value=KeyValueParameter("second_value", Any, second_value),
        )

    def execute(self):
        self.column = self.column.to_numpy()
        self.column = pd.Series(
            numpy.where(
                self.column,
                self.parameters.get_dict().get("first_value"),
                self.parameters.get_dict().get("second_value"),
            )
        )
