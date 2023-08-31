from typing import List

import click
from qwak.clients.model_management import ModelsManagementClient
from qwak.tools.logger.logger import get_qwak_logger

from qwak_sdk.commands.models.deployments.deploy._logic.deploy_config import (
    DeployConfig,
)
from qwak_sdk.commands.models.deployments.undeploy._logic.request_undeploy import (
    undeploy,
)
from qwak_sdk.inner.tools.cli_tools import QwakCommand
from qwak_sdk.inner.tools.config_handler import config_handler

logger = get_qwak_logger()


@click.command(
    name="undeploy",
    help="Model undeploy operation",
    cls=QwakCommand,
)
@click.option("--model-id", metavar="NAME", required=True, help="Model ID")
@click.option(
    "--variation-name",
    required=False,
    type=str,
    help="The variation name",
)
@click.option(
    "--environment-name",
    required=False,
    type=str,
    help="Environment to undeploy from (if not specified uses your default environment",
    multiple=True,
)
@click.option(
    "-f",
    "--from-file",
    help="Undeploy by run_config file, Command arguments will overwrite any run_config.",
    required=False,
    type=click.Path(exists=True, resolve_path=True, dir_okay=False),
)
def models_undeploy(
    model_id: str,
    variation_name: str,
    environment_name: List[str],
    from_file: str = None,
    **kwargs,
):
    logger.info(f"Initiating undeployment for model '{model_id}'")
    models_management = ModelsManagementClient()
    config: DeployConfig = config_handler(
        config=DeployConfig,
        from_file=from_file,
        out_conf=False,
        sections=("realtime",),
        model_id=model_id,
        variation_name=variation_name,
        environment_name=environment_name,
    )
    model_uuid = models_management.get_model_uuid(model_id)

    undeploy(
        model_id=model_id,
        config=config,
        model_uuid=model_uuid,
    )
