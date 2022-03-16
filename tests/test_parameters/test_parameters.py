import pytest

from simple_repo.core.parameter import (
    Parameters,
    KeyValueParameter,
    SimpleParameter,
)


@pytest.fixture
def parameters() -> Parameters:
    yield Parameters(
        ratio=KeyValueParameter("ratio", float, 0.5),
        string=KeyValueParameter("aspect", str, ""),
    )


@pytest.fixture
def read_csv_parameters(parameters) -> Parameters:
    parameters.add_all_parameters(
        path=KeyValueParameter("filename_or_buffer", str, is_mandatory=True),
        delim=KeyValueParameter("delimiter", str, ","),
    )
    yield parameters


class TestParameters:
    def test_init(self, parameters):
        params = parameters
        assert hasattr(params, "ratio") and params.ratio.value == 0.5
        assert hasattr(params, "string") and params.string.value == ""

    def test_add_parameter(self, parameters):
        parameters.add_parameter("iters", KeyValueParameter("iterations", int, 8))
        assert (
            hasattr(parameters, "iters")
            and parameters.iters.value == 8
            and parameters.iters.type == int
        )

    def test_add_all(self, parameters):
        parameters.add_all_parameters(
            par1=KeyValueParameter("p1", str, is_mandatory=True),
            par2=KeyValueParameter("p2", str, ""),
        )
        assert hasattr(parameters, "par1")
        assert hasattr(parameters, "par2")

    def test_add_group(self, read_csv_parameters):
        read_csv_parameters.add_group("read_csv", keys=["path", "delim"])
        assert read_csv_parameters.path.group_name == "read_csv"
        assert read_csv_parameters.delim.group_name == "read_csv"

    def test_add_group_not_structuredparamlist(self, parameters):
        parameters.add_group("read_csv", keys=["ratio", "elements"])
        assert parameters.ratio.group_name == "read_csv"

    def test_group_all(self, read_csv_parameters):
        read_csv_parameters.group_all("read_csv")
        assert read_csv_parameters.ratio.group_name == "read_csv"
        assert read_csv_parameters.string.group_name == "read_csv"
        assert read_csv_parameters.path.group_name == "read_csv"
        assert read_csv_parameters.delim.group_name == "read_csv"

    def test_get_all(self, parameters):
        params = parameters.get_all()
        assert all(issubclass(par.__class__, SimpleParameter) for par in params)

    def test_get_all_from_group(self, read_csv_parameters):
        read_csv_parameters.add_group("read_csv", keys=["path", "delim"])
        params = read_csv_parameters.get_all_from_group("read_csv")
        assert "ratio" not in map(lambda par: par.name, params)

    def test_get_all_from_not_existing_group(self, read_csv_parameters):
        read_csv_parameters.add_group("read_csv", keys=["path", "delim"])
        params = read_csv_parameters.get_all_from_group("read_ciessevu")
        assert not params

    def test_get_dict(self, parameters):
        group_dct = parameters.get_dict()
        assert (
            "ratio" in group_dct.keys()
            and group_dct.get("ratio") == parameters.ratio.value
        )
        assert (
            "aspect" in group_dct.keys()
            and group_dct.get("aspect") == parameters.string.value
        )
        assert "elements" not in group_dct.keys()

    def test_get_dict_from_group(self, read_csv_parameters):
        read_csv_parameters.add_group("read_csv", keys=["path", "delim"])
        group_dct = read_csv_parameters.get_dict_from_group("read_csv")
        assert (
            "filename_or_buffer" in group_dct.keys()
            and group_dct.get("filename_or_buffer") == read_csv_parameters.path.value
        )
        assert (
            "delimiter" in group_dct.keys()
            and group_dct.get("delimiter") == read_csv_parameters.delim.value
        )
        assert "ratio" not in group_dct.keys()
