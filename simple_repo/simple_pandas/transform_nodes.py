import pandas

from node_structure import PandasNode, PandasParameter, PandasParameterList


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
        dataset[column] = pandas.to_timedelta(dataset[column])
        dataset[column] = dataset[column].apply(lambda elem: elem.total_seconds())
    elif column_type == "datetime":
        dataset[column] = pandas.to_datetime(dataset[column]).dt.date
    return dataset


def _filter_feature(dataset, feature):
    """
    Utility function used to manipulate a dataset and apply the given feature
    """
    try:
        dataset = _set_column_type(dataset, feature["name"], feature["type"])
    except Exception:
        pass
    try:
        dataset = _filter_column_by_value(
            dataset, feature["name"], feature["filter_value"]
        )
    except Exception:
        pass
    return dataset


class PandasColumnSelector(PandasNode):
    _parameters = {
        # A variable number of columns can be passed, but all of them must have the same structure specified here.
        # A keyword represent the keyword that can be written in the json, the value True or False tells if it is
        # mandatory or not.
        "columns": PandasParameterList(name=True, type=False)
    }

    def __init__(self, **kwargs):
        super(PandasColumnSelector, self).__init__(**kwargs)

    def _filter(self):
        """
        Method used to apply the feature preprocessing to the given dataset.
        """
        features = self._parameters.get("columns").parameters
        if "*" in [x["name"] for x in features]:
            for feature in features:
                self.dataset = _filter_feature(self.dataset, feature)
        else:
            features_list = []
            for feature in features:
                self.dataset = _filter_feature(self.dataset, feature)
                features_list.append(feature["name"])
            self.dataset = self.dataset.filter(features_list)
        return self.dataset

    def execute(self):
        self.dataset = self._filter()


class PandasPivot(PandasNode):
    _parameters = {
        "rows": PandasParameter("index", str, is_mandatory=True),
        "columns": PandasParameter("columns", str, is_mandatory=True),
        "values": PandasParameter("values", str, is_mandatory=True),
        "aggfunc": PandasParameter("aggfunc", str, param_value="mean"),
        "fill_value": PandasParameter("fill_value", int, param_value=0),
        "dropna": PandasParameter("dropna", bool, param_value=True),
        "sort": PandasParameter("sort", bool, param_value=True),
    }

    def __init__(self, **kwargs):
        super(PandasPivot, self).__init__(**kwargs)

    def execute(self):
        param_dict = self._get_params_as_dict()
        self.dataset = pandas.pivot_table(self.dataset, **param_dict)
