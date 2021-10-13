class ParameterNotFound(Exception):
    def __init__(self, msg: str):
        super(ParameterNotFound, self).__init__(msg)


class BadParameterStructure(Exception):
    def __init__(self, msg: str):
        super(BadParameterStructure, self).__init__(msg)


class DuplicatedNodeId(Exception):
    def __init__(self, msg: str):
        super(DuplicatedNodeId, self).__init__(msg)


class EdgeConnectionError(Exception):
    def __init__(self, msg: str):
        super(EdgeConnectionError, self).__init__(msg)
