from typing import List, Tuple

import pandas

from simple_repo.exception import ParametersException
from simple_repo.parameter import KeyValueParameter, Parameters
from simple_repo.simple_pandas.node_structure import PandasNode


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


class PandasColumnsFiltering(PandasNode):
    """PandasColumnsFiltering manages filtering of columns.

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
        Keep columns for which index falls withing the given range (from, to (included)).
    """

    def __init__(
        self,
        node_id: str,
        column_indexes: List[int] = None,
        column_names: List[str] = None,
        columns_like: str = None,
        columns_regex: str = None,
        columns_range: Tuple[int, int] = None,
    ):
        super(PandasColumnsFiltering, self).__init__(node_id)

        none_parameters = sum(
            parameter is not None
            for parameter in [
                column_indexes,
                column_names,
                columns_like,
                columns_regex,
                columns_range,
            ]
        )

        if none_parameters > 1:
            raise ParametersException("Filtering parameters are mutually exclusive.")

        self.parameters = Parameters(
            columns_range=KeyValueParameter("range", str, value=columns_range),
            column_indexes=KeyValueParameter("indexes", str, value=column_indexes),
            column_names=KeyValueParameter("items", str, value=column_names),
            columns_like=KeyValueParameter("like", str, value=columns_like),
            axis=KeyValueParameter("axis", str, value="columns"),
            columns_regex=KeyValueParameter("regex", str, value=columns_regex),
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
        else:
            self.dataset = self.dataset.filter(
                **self.parameters.get_dict_from_group("filter")
            )


class PandasPivot(PandasNode):
    """Transforms a DataFrame into a Pivot from the given rows, columns and values.

    Parameters
    ----------
    rows : str
        Name of the column whose values will be the rows of the pivot.
    columns : str
        Name of the column whose values will be the columns of the pivot.
    values: str
        Name of the column whose values will be the values of the pivot.
    aggfunc: str = "mean"
        Function to use for the aggregation.
    fill_value: int = 0
        Value to replace missing values with.
    dropna: bool = True
        Do not include columns whose entries are all NaN.
    sort: bool = True
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


class PandasRenameColumn(PandasNode):
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
