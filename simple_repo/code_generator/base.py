from dataclasses import dataclass
from enum import Enum

from simple_repo.code_generator.class_analyzer import get_code


class ObjectInfo:
    def __init__(self, obj_class: object, obj_type: object, code: str = ""):
        self._obj_clsname = obj_class.__name__
        self._obj_class = obj_class
        self._obj_type = obj_type
        self._code = code
        self._priority = 0

    @property
    def obj_clsname(self):
        return self._obj_clsname

    @property
    def obj_class(self):
        return self._obj_class

    @obj_class.setter
    def obj_class(self, obj_class):
        self._obj_class = obj_class

    @property
    def obj_type(self):
        return self._obj_type

    @obj_type.setter
    def obj_type(self, obj_type):
        self._obj_type = obj_type

    @property
    def code(self):
        return self._code

    def retrieve_code(self):
        self._code = get_code(self._obj_class)

    @property
    def priority(self):
        return self._priority

    @priority.setter
    def priority(self, priority):
        self._priority = priority

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, ObjectInfo):
            return False
        return self._obj_class == o.obj_class

    def __ne__(self, o: object) -> bool:
        return not self.__eq__(o)

    def __hash__(self) -> int:
        return hash(self._obj_class)
