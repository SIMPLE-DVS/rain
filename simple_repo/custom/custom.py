from simple_repo.base import ComputationalNode


class CustomNode(ComputationalNode):
    def __init__(self, node_id: str, use_function, **kwargs):
        super(CustomNode, self).__init__(node_id)
        self._input_vars = {}
        self._output_vars = {}
        self._function = use_function
        self._other_params = kwargs

    def execute(self):
        my_vars = vars(self)
        in_vars = {inp: my_vars[inp] for inp in self._input_vars}
        self._function(in_vars, self._output_vars, **self._other_params)
        for outname, out in self._output_vars.items():
            setattr(self, outname, out)

    def __gt__(self, other):
        if isinstance(other, CustomNode):
            for outname, out in self._output_vars.items():
                other.set_input_value(outname, out)

        return super(CustomNode, self).__gt__(other)

    def __matmul__(self, other):
        if isinstance(other, str):
            self.set_input_value(other, None)
        elif isinstance(other, list):
            for inp in other:
                self.set_input_value(inp, None)

        return super(CustomNode, self).__matmul__(other)

