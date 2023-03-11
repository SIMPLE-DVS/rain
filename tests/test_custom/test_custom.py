"""
 Copyright (C) 2023 Università degli Studi di Camerino and Sigma S.p.A.
 Authors: Alessandro Antinori, Rosario Capparuccia, Riccardo Coltrinari, Flavio Corradini, Marco Piangerelli, Barbara Re, Marco Scarpetta

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU Affero General Public License as
 published by the Free Software Foundation, either version 3 of the
 License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Affero General Public License for more details.

 You should have received a copy of the GNU Affero General Public License
 along with this program.  If not, see <https://www.gnu.org/licenses/>.
 """

from rain import DataFlow, IrisDatasetLoader, PandasRenameColumn
from rain.nodes.custom import CustomNode, parse_custom_node


def sumelements(input_dict, output_dict):
    output_dict["sum"] = input_dict["dataset"]["a"].sum()


def divide(input_dict, output_dict):
    output_dict["division"] = input_dict["sum"] / 5


def function_test(input_dict, output_dict, test, test1, test2=True, test3="ciao"):
    a = [test, test1, test2, test3]
    output_dict["sum"] = input_dict["dataset"]["a"].sum()
    output_dict["fail?"] = input_dict[",.fail2"].get()
    a.reverse()
    output_dict["test_out"]["b"] = input_dict.get("test_in")


def test_custom_node():
    df = DataFlow("df1")

    iris = IrisDatasetLoader("loadiris")
    rename = PandasRenameColumn("rcol", ["a", "b", "c", "d"])
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
