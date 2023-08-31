from qwak_inference.constants import QwakConstants


class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Session(metaclass=SingletonMeta):
    __environment = QwakConstants.QWAK_DEFAULT_SECTION

    def set_environment(self, environment):
        self.__environment = environment

    def get_environment(self):
        return self.__environment
