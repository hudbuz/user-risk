import contextlib
import logging
import os
import platform
import shutil
import sys
import uuid
from pathlib import Path

from qwak_sdk import __version__ as qwak_sdk_version
from qwak_sdk.commands.models.build._logic.config.config_v1 import ConfigV1
from qwak_sdk.commands.models.build._logic.constant.host_resource import (
    HOST_QWAK_HIDDEN_FOLDER,
)
from qwak_sdk.inner.tools.logger import setup_qwak_logger
from qwak_sdk.inner.tools.logger.logger import (
    BUILD_LOCAL_FILE_HANDLER_NAME,
    BUILD_LOCAL_LOGGER_NAME,
    CONSOLE_HANDLER_NAME,
    REMOTE_CONSOLE_HANDLER_NAME,
    REMOTE_LOGGER_NAME,
    VERBOSITY_LEVEL_MAPPING,
    get_qwak_logger,
    non_qwak_logger_enabled,
    set_file_handler_log_file,
    set_handler_verbosity,
)

BUILDS_LOGS = HOST_QWAK_HIDDEN_FOLDER / "logs" / "build"
BUILD_LOG_NAME = "build.log"
MAX_LOGS_NUMBER = 15


@contextlib.contextmanager
def build_logger(config: ConfigV1, json_logs: bool):
    log_path = BUILDS_LOGS / config.build_properties.model_id / str(uuid.uuid4())[:4]
    log_path.mkdir(parents=True, exist_ok=True)
    try:
        (log_path / "build_config.yml").write_text(config.to_yaml())
        log_system_information(log_path)

        log_file = log_path / BUILD_LOG_NAME
        setup_qwak_logger()
        yield setup_logger(
            log_file=log_file, verbosity_level=config.verbose, json_logs=json_logs
        ), log_path
    finally:
        # Cleanup - Save only x last zips
        logs_zip_sorted_by_data = sorted(
            BUILDS_LOGS.rglob("**/*"), key=os.path.getmtime
        )[:-MAX_LOGS_NUMBER]
        path: Path
        for path in logs_zip_sorted_by_data:
            if path.is_file():
                os.remove(path)
            elif path.is_dir():
                shutil.rmtree(path, ignore_errors=True)


def setup_logger(
    log_file: Path, verbosity_level: int, json_logs: bool
) -> logging.Logger:
    # Init logger
    fallback_logger_name = (
        BUILD_LOCAL_LOGGER_NAME if not json_logs else REMOTE_LOGGER_NAME
    )
    logger = get_qwak_logger(fallback_logger_name=fallback_logger_name)

    if not non_qwak_logger_enabled():
        if logger.name == BUILD_LOCAL_LOGGER_NAME:
            set_file_handler_log_file(logger, BUILD_LOCAL_FILE_HANDLER_NAME, log_file)
            set_handler_verbosity(
                logger, CONSOLE_HANDLER_NAME, VERBOSITY_LEVEL_MAPPING[verbosity_level]
            )
        elif logger.name == REMOTE_LOGGER_NAME and json_logs:
            set_handler_verbosity(
                logger,
                REMOTE_CONSOLE_HANDLER_NAME,
                VERBOSITY_LEVEL_MAPPING[verbosity_level],
            )

    return logger


def log_system_information(destination: Path):
    (destination / "python_version").write_text(sys.version)
    (destination / "qwak_sdk_version").write_text(qwak_sdk_version)
    (destination / "os_detail").write_text(platform.platform())
