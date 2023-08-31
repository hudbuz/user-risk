import click
from qwak.inner.const import QwakConstants
from qwak.tools.logger.logger import set_qwak_logger_stdout_verbosity_level

from qwak_sdk.commands.models.deployments.deploy._logic.deploy_config import (
    DeployConfig,
    PurchaseOption,
)
from qwak_sdk.commands.models.deployments.deploy._logic.deployment_response_handler import (
    client_deployment,
)
from qwak_sdk.commands.models.deployments.deploy.batch._logic.deploy_executor import (
    BatchDeployExecutor,
)
from qwak_sdk.inner.tools.cli_tools import DeprecatedOption, QwakCommand
from qwak_sdk.inner.tools.config_handler import config_handler


@click.command("batch", help="Deploy batch model.", cls=QwakCommand)
@click.option(
    "--build-id",
    help="Build ID, If not specified latest successful build will deployed.",
    type=str,
)
@click.option("--model-id", help="Model ID", type=str, required=True)
@click.option(
    "--pods",
    "--replicas",
    type=int,
    help="Number of pods to deploy.",
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
    help="Indicate the instance deployments type, whether it is spot/ondemand.  Default is spot",
)
@click.option(
    "--instance",
    required=False,
    type=str,
    help="The instance size to deploy on - 'small', 'medium', 'large', etc...",
    default=None,
)
def batch(verbose: bool, from_file: str, out_conf: bool, sync: bool, **kwargs):
    set_qwak_logger_stdout_verbosity_level(verbose + 1)
    config: DeployConfig = config_handler(
        config=DeployConfig,
        from_file=from_file,
        out_conf=out_conf,
        sections=("batch",),
        **kwargs,
    )
    if not out_conf:
        deploy_executor = BatchDeployExecutor(config)
        client_deployment(deploy=deploy_executor, sync=sync)
