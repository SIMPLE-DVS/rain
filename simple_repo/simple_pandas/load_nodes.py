import pandas

from node_structure import PandasNode
from simple_repo.parameter import KeyValueParameter


class PandasCSVLoader(PandasNode):
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

    _parameters = {
        "path": KeyValueParameter("filepath_or_buffer", str, is_mandatory=True),
        "delim": KeyValueParameter("delimiter", str, is_mandatory=True),
    }

    def __init__(self, **kwargs):
        super(PandasCSVLoader, self).__init__(**kwargs)

    def execute(self):
        param_dict = self._get_params_as_dict()
        self.dataset = pandas.read_csv(**param_dict)


if __name__ == "__main__":
    loader = PandasCSVLoader(path="C:/Users/RICCARDO/Desktop/iris_ds.csv")
    loader.execute()
    print(loader.dataset)
