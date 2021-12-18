from simple_repo import DataFlow, PandasRenameColumn, PandasIrisLoader
from simple_repo.execution import LocalExecutor


class TestExecutors:
    def test_local_executor(self):
        df = DataFlow("dataflow1")
        load = PandasIrisLoader("iris")
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
