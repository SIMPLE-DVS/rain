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

"""
Module used to manage all the Exception that could occur during the creation and execution of a Dataflow
"""


class DuplicatedNodeId(Exception):
    def __init__(self, msg: str):
        super(DuplicatedNodeId, self).__init__(msg)


class EdgeConnectionError(Exception):
    def __init__(self, msg: str):
        super(EdgeConnectionError, self).__init__(msg)


class CyclicDataFlowException(Exception):
    def __init__(self, dataflow_id: str):
        super(CyclicDataFlowException, self).__init__(
            "DataFlow {} has cycles.".format(dataflow_id)
        )


class ParametersException(ValueError):
    def __init__(self, msg):
        super(ParametersException, self).__init__(msg)


class PandasSequenceException(Exception):
    def __init__(self, msg):
        super(PandasSequenceException, self).__init__(msg)


class EstimatorNotFoundException(Exception):
    def __init__(self, msg):
        super(EstimatorNotFoundException, self).__init__(msg)


class InputNotFoundException(Exception):
    def __init__(self, msg):
        super(EstimatorNotFoundException, self).__init__(msg)
