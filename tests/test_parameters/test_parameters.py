import pytest

from simple_repo.exception import BadParameterStructure, ParameterNotFound
from simple_repo.parameter import (
    Parameters,
    KeyValueParameter,
    StructuredParameterList,
    SimpleParameter,
)


@pytest.fixture
def parameters() -> Parameters:
    yield Parameters(
        ratio=KeyValueParameter("ratio", float, 0.5),
        string=KeyValueParameter("aspect", str, ""),
        elements=StructuredParameterList(lorem=True, ipsum=False),
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
        assert hasattr(params, "elements") and isinstance(
            params.elements, StructuredParameterList
        )

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
        assert parameters.elements.group_name is None

    def test_group_all(self, read_csv_parameters):
        read_csv_parameters.group_all("read_csv")
        assert read_csv_parameters.ratio.group_name == "read_csv"
        assert read_csv_parameters.string.group_name == "read_csv"
        assert read_csv_parameters.elements.group_name is None
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


class TestStructuredParameterList:
    @pytest.fixture
    def correct_param(self):
        yield [
            {"name": "Sherlock Holmes", "address": "221B Baker Street"},
            {"name": "Sweeney Todd", "address": "Fleet Street"},
            {"name": "Rick Deckard"},
        ]

    def test_bad_param_structure(self):
        with pytest.raises(BadParameterStructure):
            StructuredParameterList(name="n", val=5)

    def test_add_parameter(self, correct_param):
        spl = StructuredParameterList(name=True, address=False)
        spl.add_parameter(**correct_param[0])

        assert spl.has_parameters(name="Sherlock Holmes")

    def test_paramnotfound_mandatories(self):
        spl = StructuredParameterList(name=True, address=False)

        with pytest.raises(ParameterNotFound):
            spl.add_parameter(address="Fleet Street")

    def test_paramnotfound_optionals(self, correct_param):
        spl = StructuredParameterList(name=True, address=False)
        param = correct_param[0]
        param["location"] = "England"
        with pytest.raises(ParameterNotFound):
            spl.add_parameter(**param)

    def test_add_parameters(self, correct_param):
        spl = StructuredParameterList(name=True, address=False)
        spl.add_all_parameters(*correct_param)

        assert (
            spl.has_parameters(name="Sherlock Holmes", address="221B Baker Street")
            and spl.has_parameters(name="Sweeney Todd", address="Fleet Street")
            and spl.has_parameters(name="Rick Deckard")
        )

    def test_paramnotfound_mandatories_in_list(self, correct_param):
        spl = StructuredParameterList(name=True, address=False)
        correct_param.append({"address": "698 Candlewood Lane"})
        with pytest.raises(ParameterNotFound):
            spl.add_all_parameters(*correct_param)

    def test_paramnotfound_optionals_in_list(self, correct_param):
        spl = StructuredParameterList(name=True, address=False)
        correct_param.append({"name": "Jessica Fletcher", "location": "Maine"})
        with pytest.raises(ParameterNotFound):
            spl.add_all_parameters(*correct_param)
