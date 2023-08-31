from qwak.inner.const import QwakConstants
from qwak.inner.singleton_meta import SingletonMeta


class Session(metaclass=SingletonMeta):
    __environment = QwakConstants.QWAK_DEFAULT_SECTION

    def set_environment(self, environment):
        self.__environment = environment

    def get_environment(self):
        return self.__environment
