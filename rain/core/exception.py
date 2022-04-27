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
