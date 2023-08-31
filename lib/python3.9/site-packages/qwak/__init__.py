"""Top-level package for qwak-core."""

__author__ = "Qwak.ai"
__version__ = "0.0.205"

from qwak.model.experiment_tracking import log_metric, log_param
from qwak.model_loggers.artifact_logger import load_file, log_file
from qwak.model_loggers.data_logger import load_data, log_data
from qwak.model_loggers.model_logger import load_model, log_model

from .model.decorators.api import api_decorator as api
from .model.decorators.timer import qwak_timer
from .qwak_client.client import QwakClient

from qwak.inner.di_configuration import wire_dependencies

_container = wire_dependencies()
