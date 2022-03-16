from rain.core.base import ComputationalNode, Tags, LibTag, TypeTag
import inspect
import re


class CustomNode(ComputationalNode):
    """A node that can contain user-defined Python code."""

    def __init__(self, node_id: str, use_function, **kwargs):
        super(CustomNode, self).__init__(node_id)
        inputs, outputs, self._other_params = parse_custom_node(use_function)

        self._input_vars = {inp: None for inp in inputs}
        self._output_vars = {out: None for out in outputs}

        for var in set(self._input_vars).union(self._output_vars):
            setattr(self, var, None)

        self._function = use_function

    def execute(self):
        my_vars = vars(self)
        in_vars = {inp: my_vars[inp] for inp in self._input_vars}
        self._function(in_vars, self._output_vars, **self._other_params)
        for outname, out in self._output_vars.items():
            setattr(self, outname, out)

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.CUSTOM, TypeTag.CUSTOM)


def parse_custom_node(custom_function):
    """Given a function, returns the inputs, outputs and kwargs that the corresponding CustomNode should use"""

    params = list(inspect.signature(custom_function).parameters.values())
    kwargs_dict = get_kwargs(params)

    code = inspect.getsource(custom_function)
    inputs = get_variables_matches(
        code, r"{}(?:\[|\.get\()\"([a-zA-Z_\d-]+)\"".format(params[0])
    )
    outputs = get_variables_matches(
        code, r"{}\[\"([a-zA-Z_\d-]+)\"\]".format(params[1])
    )

    return inputs, outputs, kwargs_dict


def get_variables_matches(code, regex):
    matches = re.findall(regex, code, re.MULTILINE)
    return matches


def get_kwargs(params):
    kwargs_dict = {}
    for p in params[2:]:
        if p.default is inspect.Signature.empty:
            kwargs_dict[p.name] = None
        else:
            kwargs_dict[p.name] = p.default
    return kwargs_dict
