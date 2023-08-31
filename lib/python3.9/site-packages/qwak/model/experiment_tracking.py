from typing import Dict, Union


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class QwakExperimentTracking(metaclass=SingletonMeta):
    def __init__(self):
        self.metric = {}
        self.param = {}

    def log_param(self, params: Dict[str, Union[int, str]]):
        self.param = {**self.param, **params}

    def log_params(self, params: Dict[str, Union[int, str]]):
        self.log_param(params)

    def log_metric(self, metrics: Dict[str, Union[int, str]]):
        self.metric = {**self.metric, **metrics}

    def log_metrics(self, metrics: Dict[str, Union[int, str]]):
        self.log_metric(metrics)

    def get_metrics(self):
        return self.metric

    def get_params(self):
        return self.param


def log_metric(metrics):
    client = QwakExperimentTracking()
    client.log_metric(metrics)


def log_param(params):
    client = QwakExperimentTracking()
    client.log_param(params)


def log_metrics(metrics):
    client = QwakExperimentTracking()
    client.log_metric(metrics)


def log_params(params):
    client = QwakExperimentTracking()
    client.log_param(params)


def get_params():
    client = QwakExperimentTracking()
    return client.get_params()


def get_metrics():
    client = QwakExperimentTracking()
    return client.get_metrics()
