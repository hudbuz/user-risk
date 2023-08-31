from typing import Optional

import click
from qwak.inner.const import QwakConstants
from qwak.tools.logger.logger import (
    get_qwak_logger,
    set_qwak_logger_stdout_verbosity_level,
)

from qwak_sdk.commands.models.deployments.deploy._logic.deploy_config import (
    DeployConfig,
    PurchaseOption,
)
from qwak_sdk.commands.models.deployments.deploy._logic.deployment_response_handler import (
    client_deployment,
)
from qwak_sdk.commands.models.deployments.deploy.realtime._logic.deploy_executor import (
    RealtimeDeployExecutor,
)
from qwak_sdk.inner.tools.cli_tools import DeprecatedOption, QwakCommand
from qwak_sdk.inner.tools.config_handler import config_handler

logger = get_qwak_logger()


@click.command("realtime", help="Deploy realtime model.", cls=QwakCommand)
@click.option(
    "--build-id",
    help="Build ID, If not specified latest successful build will deployed.",
    type=str,
)
@click.option("--model-id", help="Model ID", type=str, required=True)
@click.option(
    "--timeout",
    type=int,
    help="Inference request timeout in MS, (e.g. 2000).",
)
@click.option(
    "--server-workers",
    type=int,
    help="Number of workers of http server. (e.g. 4).",
)
@click.option(
    "--daemon-mode/--no-daemon-mode",
    type=bool,
    default=True,
    help="Configure Gunicorn daemon mode. Default is True",
)
@click.option(
    "--pods",
    "--replicas",
    type=int,
    help="Number of replicas to deploy.",
    deprecated=["--pods"],
    preferred="--replicas",
    cls=DeprecatedOption,
)
@click.option(
    "--cpus",
    type=float,
    help="Number of CPU cores, can be fraction of a core (e.g. 0.4).",
)
@click.option(
    "--memory",
    type=int,
    help="Memory amount in Mib (e.g. 256).",
)
@click.option(
    "--gpu-type",
    metavar="NAME",
    required=False,
    help=f"Type of GPU to use on the deployments ({', '.join([x for x in QwakConstants.GPU_TYPES])}).",
    type=click.STRING,
)
@click.option(
    "--gpu-amount",
    metavar="NAME",
    required=False,
    type=int,
    help="Amount of GPU to use on the deployments.",
)
@click.option(
    "--iam-role-arn",
    type=str,
    help="Custom IAM Role ARN.",
)
@click.option(
    "--max-batch-size",
    type=int,
    help="Max batch size in prediction. (defaults to 0 which is dynamic)",
)
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
    help="Environment to deploy on (if not specified uses your default environment",
    multiple=True,
)
@click.option(
    "--sync",
    is_flag=True,
    default=False,
    help="Waiting for deployments to be ready",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Log verbosity level - v: INFO, vv: DEBUG [default: WARNING]",
)
@click.option(
    "-f",
    "--from-file",
    help="Deploy by run_config file, Command arguments will overwrite any run_config.",
    required=False,
    type=click.Path(exists=True, resolve_path=True, dir_okay=False),
)
@click.option(
    "--out-conf",
    help="Extract deploy conf from commands arguments, the commands will not run it wil only output valid yaml "
    "structure",
    default=False,
    is_flag=True,
)
@click.option(
    "--purchase-option",
    required=False,
    type=click.Choice([i.value for i in PurchaseOption], case_sensitive=False),
    help="Indicate the instance deployments type, whether it is spot/ondemand. Default is spot",
)
@click.option(
    "--deployment-timeout",
    type=int,
    help="The number of seconds the deployments can be in progress before it is considered as failed. This is useful "
    "in cases where a deployments has a high number of pods to replace, scarce resources (gpu), or large build "
    "size. Default is 1800 seconds (30 minutes).",
)
@click.option(
    "-E",
    "--env-vars",
    required=False,
    metavar="NAME=VALUE",
    multiple=True,
    help="A parameter for the deployments, of the form -E name=value",
)
@click.option(
    "--instance",
    required=False,
    type=str,
    help="The instance size to deploy on - 'small', 'medium', 'large', etc...",
    default=None,
)
@click.option(
    "--protected",
    "variation_protected_state",
    required=False,
    type=bool,
    default=False,
    help="Whether the deployment variation is protected . Default is False",
)
def realtime(
    verbose: bool, from_file: Optional[str], out_conf: bool, sync: bool, **kwargs
):
    set_qwak_logger_stdout_verbosity_level(verbose + 1)
    deploy_realtime(from_file, out_conf, sync, **kwargs)


def deploy_realtime(from_file: Optional[str], out_conf: bool, sync: bool, **kwargs):
    config: DeployConfig = config_handler(
        config=DeployConfig,
        from_file=from_file,
        out_conf=out_conf,
        sections=("realtime", "autoscaling"),
        **kwargs,
    )
    if not out_conf:
        deploy_executor = RealtimeDeployExecutor(config)
        client_deployment(deploy=deploy_executor, sync=sync)
