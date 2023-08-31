from typing import Tuple

import click
from qwak.exceptions import QwakException
from qwak.tools.logger.logger import get_qwak_logger

from qwak_sdk.commands.models.runtime.traffic_update._logic.execute_runtime_update_traffic import (
    execute_runtime_traffic_update,
)
from qwak_sdk.inner.tools.cli_tools import QwakCommand

logger = get_qwak_logger()


@click.command("traffic_update", cls=QwakCommand)
@click.option("-m", "--model-id", required=True, help="Model named ID")
@click.option(
    "--environment-name",
    required=False,
    type=str,
    help="Environments to deploy on (if not specified uses your default environment)",
    multiple=True,
)
@click.option(
    "--from-file",
    required=True,
    help="The variations config file path",
    type=click.Path(exists=True, resolve_path=True, dir_okay=False),
)
def runtime_traffic_update(
    model_id: str, from_file: str, environment_name: Tuple[str], **kwargs
):
    try:
        execute_runtime_traffic_update(model_id, from_file, environment_name)
        logger.info(f"Successfully updated traffic for models {model_id}")
    except Exception as e:
        raise QwakException(f'Failed to apply traffic configurations. Error is "{e}"')
