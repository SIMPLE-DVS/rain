import simple_repo
from simple_repo.custom.custom import CustomNode, parse_custom_node


def sumelements(input_dict, output_dict):
    output_dict["sum"] = input_dict["dataset"]["a"].sum()


def divide(input_dict, output_dict):
    output_dict["division"] = input_dict["sum"] / 5


def function_test(input_dict, output_dict, test, test1, test2=True, test3="ciao"):
    a = [test, test1, test2, test3]
    output_dict["sum"] = input_dict["dataset"]["a"].sum()
    output_dict["fail?"] = input_dict[",.fail2"].get()
    a.reverse()
    output_dict["test_out"]["b"] = input_dict["test_in"]


def test_custom_node():
    df = simple_repo.DataFlow("df1")

    iris = simple_repo.PandasIrisLoader("loadiris")
    rename = simple_repo.PandasRenameColumn("rcol", ["a", "b", "c", "d"])
    cnode = CustomNode("c", use_function=sumelements)
    cnode2 = CustomNode("c2", use_function=divide)

    df.add_edges(
        [
            iris @ "dataset" > rename @ "dataset",
            rename @ "dataset" > cnode @ "dataset",
            cnode @ "sum" > cnode2 @ "sum",
        ]
    )

    df.execute()

    print(cnode2.division)


def test_parse_custom_node():
    inputs, outputs, kwargs = parse_custom_node(function_test)
    assert inputs == ["dataset", "test_in"]
    assert outputs == ["sum", "test_out"]
    assert kwargs == {"test": None, "test1": None, "test2": True, "test3": "ciao"}
