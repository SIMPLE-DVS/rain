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

from rain.core.base import ComputationalNode, Tags, LibTag, TypeTag
import inspect
import re


class CustomNode(ComputationalNode):
    """A node that can contain user-defined Python code."""

    def __init__(self, node_id: str, use_function, **kwargs):
        super(CustomNode, self).__init__(node_id)
        inputs, outputs, self._other_params = parse_custom_node(use_function)
        for k, v in kwargs.items():
            self._other_params[k] = v

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
        return Tags(LibTag.BASE, TypeTag.CUSTOM)


def parse_custom_node(custom_function):
    """Given a function, returns the inputs, outputs and kwargs that the corresponding CustomNode should use"""

    params = list(inspect.signature(custom_function).parameters.values())
    kwargs_dict = get_kwargs(params)

    code = inspect.getsource(custom_function)
    inputs = get_variables_matches(
        code, r"{}(\[|\.get\()(\"|\')(?P<param>[a-zA-Z_\d-]+)(\"|\')(\]|\))".format(params[0])
    )
    outputs = get_variables_matches(
        code, r"{}\[(\"|\')(?P<param>[a-zA-Z_\d-]+)(\"|\')\]".format(params[1])
    )

    return inputs, outputs, kwargs_dict


def get_variables_matches(code, regex):
    return [x.group("param") for x in re.finditer(regex, code, re.MULTILINE)]


def get_kwargs(params):
    kwargs_dict = {}
    for p in params[2:]:
        if p.default is inspect.Signature.empty:
            kwargs_dict[p.name] = None
        else:
            kwargs_dict[p.name] = p.default
    return kwargs_dict
