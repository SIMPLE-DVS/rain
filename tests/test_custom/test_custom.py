import simple_repo
from simple_repo.custom.custom import CustomNode


def sumelements(input_dict, output_dict):
    output_dict["sum"] = input_dict["dataset"]["a"].sum()


def divide(input_dict, output_dict):
    output_dict["division"] = input_dict["sum"] / 5


def test_custom_node():
    df = simple_repo.DataFlow("df1")

    iris = simple_repo.PandasIrisLoader("loadiris")
    rename = simple_repo.PandasRenameColumn("rcol", ["a", "b", "c", "d"])
    cnode = CustomNode("c", use_function=sumelements)
    cnode2 = CustomNode("c2", use_function=divide)

    df.add_edges([
        iris > rename,
        rename @ "dataset" > cnode,
        cnode @ "sum" > cnode2
    ])

    df.execute()

    print(cnode2.division)
