class ParameterNotFound(Exception):
    def __init__(self, msg: str):
        super(ParameterNotFound, self).__init__(msg)


class BadParameterStructure(Exception):
    def __init__(self, msg: str):
        super(BadParameterStructure, self).__init__(msg)


class BadSimpleNodeClass(Exception):
    def __init__(self, msg: str):
        super(BadSimpleNodeClass, self).__init__(msg)


class BadSimpleModule(Exception):
    def __init__(self, msg: str):
        super(BadSimpleModule, self).__init__(msg)


class BadSimpleParameterName(Exception):
    def __init__(self, msg: str):
        super(BadSimpleParameterName, self).__init__(msg)


class BadSimpleParameterType(Exception):
    def __init__(self, msg: str):
        super(BadSimpleParameterType, self).__init__(msg)


class MissingMandatoryParameter(Exception):
    def __init__(self, msg: str):
        super(MissingMandatoryParameter, self).__init__(msg)


class UnexpectedParameter(Exception):
    def __init__(self, msg: str):
        super(UnexpectedParameter, self).__init__(msg)


class MissingSimpleNodeKey(Exception):
    def __init__(self, msg: str):
        super(MissingSimpleNodeKey, self).__init__(msg)
