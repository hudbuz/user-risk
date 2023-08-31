import logging
import logging.config
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import yaml
from qwak.exceptions import QwakException
from qwak.inner.const import QwakConstants

DEFAULT_LOGGER_NAME = "qwak"
REMOTE_LOGGER_NAME = "qwak_remote"
MODEL_LOGGER_NAME = "qwak_model"
FEATURE_STORE_LOGGER_NAME = "feature_store"
BUILD_LOCAL_LOGGER_NAME = "build_local"
DOCKER_INTERNAL_LOGGER_NAME = "docker_internal"

ENVIRON_LOGGER_TYPE = "LOGGER_TYPE"

BUILD_LOCAL_FILE_HANDLER_NAME = "build_log_file_handler"
FILE_HANDLER_NAME = "file_handler"
CONSOLE_HANDLER_NAME = "console"
REMOTE_CONSOLE_HANDLER_NAME = "remote_console"

DEFINED_LOGGER_NAMES = {
    DEFAULT_LOGGER_NAME,
    REMOTE_LOGGER_NAME,
    MODEL_LOGGER_NAME,
    FEATURE_STORE_LOGGER_NAME,
    BUILD_LOCAL_LOGGER_NAME,
    DOCKER_INTERNAL_LOGGER_NAME,
}

VERBOSITY_LEVEL_MAPPING = {0: logging.CRITICAL, 1: logging.INFO, 2: logging.DEBUG}
REVERSED_VERBOSITY_LEVEL_MAPPING = {
    value: key for key, value in VERBOSITY_LEVEL_MAPPING.items()
}

AIRFLOW_ENV_FLAG = "AIRFLOW__LOGGING__REMOTE_LOGGING"


def setup_qwak_logger(
    default_level: int = logging.INFO,
    logger_name_handler_addition: str = None,
    disable_existing_loggers: bool = False,
):
    """Setup qwak logger:
            1. Rotating file in $HOME/.qwak/log/sdk.log (10MB * 5 files) (DEBUG level)
            2. Stdout logger with colored logs. (INFO level)

    Args:
        default_level: Default logging level in case of failure
        logger_name_handler_addition:
            Logger name which the handlers of will be appended to all loggers which have handlers
            Overriding stdout stream handler if exists
        disable_existing_loggers: disables all existing loggers

    Raises:
        QwakException: If loading logging.yml fails or the preparation of the logging environment raises an exceptions

    Notes:
        1. https://docs.python.org/3/library/logging.html#logging-levels
    """

    config_file = Path(__file__).parent / "logging.yml"
    if config_file.exists() and not non_qwak_logger_enabled():
        with config_file.open(mode="rt") as f:
            try:
                # Creating log directory
                log_path = Path(QwakConstants.QWAK_CONFIG_FOLDER)
                log_path.mkdir(parents=True, exist_ok=True)
                # Load logger configuration
                config = yaml.safe_load(f.read())
                config["handlers"][FILE_HANDLER_NAME]["filename"] = str(
                    log_path / "sdk.log"
                )
                config["handlers"][BUILD_LOCAL_FILE_HANDLER_NAME]["filename"] = str(
                    log_path / "sdk.log"
                )
                config["disable_existing_loggers"] = disable_existing_loggers

                logging.config.dictConfig(config)

                if logger_name_handler_addition:
                    if (
                        logger_name_handler_addition
                        in logging.Logger.manager.loggerDict
                    ):
                        _add_logger_handlers(logger_name_handler_addition)
                    else:
                        print(
                            "Tried to set orphan loggers handlers with a non-existing logger name handlers'"
                        )

            except Exception as e:
                raise QwakException(
                    f"Error in Logging Configuration. Error message: {e}"
                )
    else:
        logging.basicConfig(level=default_level)
        print("Failed to load configuration file. Using default configs")


def _add_logger_handlers(logger_name):
    """
    Add a specific logger handlers to all loggers
    Override loggers StreamHandler handlers if the input logger has a StreamHandler

    Args:
        logger_name: logger name which consists of the handlers we wish to set
    """

    logger_handlers = logging.getLogger(logger_name).handlers
    base_contains_stdout = any(
        [
            handler
            for handler in logger_handlers
            if type(handler) is logging.StreamHandler
        ]
    )

    for logger in [
        logger
        for logger in logging.Logger.manager.loggerDict.values()
        if not isinstance(logger, logging.PlaceHolder)
    ]:
        if logger.handlers:
            logger.handlers = [
                handler
                for handler in logger.handlers
                if base_contains_stdout and type(handler) is not logging.StreamHandler
            ] + logger_handlers


def get_qwak_logger(
    logger_name: Optional[str] = None, fallback_logger_name: Optional[str] = None
) -> logging.Logger:
    """Get qwak logger (Singleton)

    Returns:
        logging.Logger: Qwak logger.
    """
    if not logger_name:
        logger_name = get_qwak_logger_name(fallback_logger_name)

    if (logger_name not in DEFINED_LOGGER_NAMES) and not non_qwak_logger_enabled:
        print("Failed to get requested logger name. Using default logger")

    return logging.getLogger(logger_name)


def set_qwak_logger_stdout_verbosity_level(verbose: int, format: str = "text"):
    """Set qwak stdout to verbose (a.k.a DEBUG level)

    Args:
        verbose: Log verbosity level - 0: WARNING, 1:INFO, 2: DEBUG

    Notes:
        1. https://docs.python.org/3/library/logging.html#logging-levels
    """
    if format == "json":
        verbose = 0
    logger: logging.Logger = get_qwak_logger()
    logger.setLevel(VERBOSITY_LEVEL_MAPPING[verbose])
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(VERBOSITY_LEVEL_MAPPING[verbose])


def get_qwak_logger_verbosity_level() -> Optional[int]:
    """Get current Qwak logger level.

    Returns:
        int: Qwak logger level 10 < level < 50.

    Notes:
        1. https://docs.python.org/3/library/logging.html#logging-levels
        2. when we update the log level through set_qwak_logger_stdout_verbosity_level we update all handler levels
           thus returning the first stream handler should be correct
    """

    logger: logging.Logger = get_qwak_logger()

    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            return logging.getLevelName(handler.level)


def set_qwak_logger_propagate(prop: bool):
    """Set qwak logger propagation

    Args:
        prop: True if propagate else False

    """
    get_qwak_logger().propagate = prop


def translate_to_qwak_verbosity_level(loglevel: int) -> Optional[int]:
    """Get qwak equivalent of the logging verbosity levels

    Args:
        loglevel: logging log level int
    """
    return REVERSED_VERBOSITY_LEVEL_MAPPING.get(loglevel, None)


def get_qwak_logger_name(fallback_logger_name: Optional[str] = None):
    if fallback_logger_name:
        return os.getenv(ENVIRON_LOGGER_TYPE, fallback_logger_name)
    else:
        return os.getenv(ENVIRON_LOGGER_TYPE, DEFAULT_LOGGER_NAME)


def get_handler_from_logger(
    logger: logging.Logger, handler_name: str
) -> logging.Handler:
    matching_handlers = list(
        filter(lambda h: h.get_name() == handler_name, logger.handlers)
    )
    if len(matching_handlers) == 0 and not non_qwak_logger_enabled:
        raise QwakException(
            f"Error in setting log file. Error message: handler of name {handler_name} was not found in logger"
        )
    elif len(matching_handlers) > 1:
        raise QwakException(
            f"Error in setting log file. Error message: handler of name {handler_name} was found more than once "
            f"in logger"
        )

    return matching_handlers[0]


def set_file_handler_log_file(
    logger: logging.Logger, handler_name: str, log_file: Path
):
    existing_handler = get_handler_from_logger(logger, handler_name)
    if type(existing_handler) != RotatingFileHandler:
        raise QwakException(
            f"Error in setting log file. Error message: handler of name {handler_name} is not a file logger handler"
        )
    replacement_handler: RotatingFileHandler = copy_file_handler_from_existing(
        existing_handler, log_file
    )
    logger.removeHandler(existing_handler)
    logger.addHandler(replacement_handler)


def copy_file_handler_from_existing(
    handler: RotatingFileHandler, log_file: Path
) -> RotatingFileHandler:
    return RotatingFileHandler(
        log_file,
        mode=handler.mode,
        maxBytes=int(handler.maxBytes),
        backupCount=int(handler.backupCount),
        encoding=handler.encoding,
        delay=handler.delay,
    )


def set_handler_verbosity(logger: logging.Logger, handler_name: str, log_level: int):
    handler = get_handler_from_logger(logger, handler_name)
    handler.setLevel(log_level)


def non_qwak_logger_enabled():
    return os.getenv(AIRFLOW_ENV_FLAG) is not None
