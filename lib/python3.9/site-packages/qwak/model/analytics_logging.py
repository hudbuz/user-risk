from qwak.inner.singleton_meta import SingletonMeta


class QwakAnalyticsLogger(metaclass=SingletonMeta):
    """
    Base class for Qwak analytics logger
    """

    def log(self, log):
        """
        Log data to Qwak's data lake
        :param log: data to be persisted
        """
        pass
