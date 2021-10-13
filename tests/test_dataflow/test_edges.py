from simple_repo import PandasCSVLoader, PandasPivot


class TestEdgesConnections:
    def test_gt(self):
        n = PandasCSVLoader("load", "./iris.csv")
        t = PandasPivot("piv", "r", "c", "v")

        edge = n > t

        intersection = set(n._output_vars.keys()).intersection(
            set(t._input_vars.keys())
        )

        assert (
            edge.source == n
            and edge.destination == t
            and all(key in n._output_vars.keys() for key in intersection)
            and all(key in t._input_vars.keys() for key in intersection)
        )
