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

from rain import DataFlow, PandasRenameColumn, IrisDatasetLoader
from rain.core.execution import LocalExecutor


class TestExecutors:
    def test_local_executor(self):
        df = DataFlow("dataflow1")
        load = IrisDatasetLoader("iris")
        rename = PandasRenameColumn(
            "rcol",
            columns=[
                "lungh. sepalo",
                "largh. sepalo",
                "lungh. petalo",
                "largh. petalo",
            ],
        )
        df.add_nodes([load, rename])
        df.add_edge(load @ "dataset" > rename @ "dataset")

        LocalExecutor().execute(df)

        assert list(rename.dataset.columns) == [
            "lungh. sepalo",
            "largh. sepalo",
            "lungh. petalo",
            "largh. petalo",
        ]
