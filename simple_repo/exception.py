class ParameterNotFound(Exception):
    def __init__(self, msg: str):
        super(ParameterNotFound, self).__init__(msg)


class BadParameterStructure(Exception):
    def __init__(self, msg: str):
        super(ParameterNotFound, self).__init__(msg)
