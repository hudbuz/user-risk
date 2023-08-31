import click
from qwak.tools.logger.logger import (
    get_qwak_logger,
    set_qwak_logger_stdout_verbosity_level,
)

from qwak_sdk.commands.models.deployments.deploy._logic.deploy_config import (
    AutoOffsetReset,
    CompressionTypes,
    DeployConfig,
    PurchaseOption,
)
from qwak_sdk.commands.models.deployments.deploy._logic.deployment_response_handler import (
    client_deployment,
)
from qwak_sdk.commands.models.deployments.deploy.streaming._logic.deploy_executor import (
    StreamDeployExecutor,
)
from qwak_sdk.inner.tools.cli_tools import DeprecatedOption, QwakCommand
from qwak_sdk.inner.tools.config_handler import config_handler

logger = get_qwak_logger()
PENDING_STATUSES = [
    "INITIATING_DEPLOYMENT",
    "PENDING_DEPLOYMENT",
    "INITIATING_UNDEPLOYMENT",
    "PENDING_UNDEPLOYMENT",
]
DEPLOYED_STATUSES = ["SUCCESSFUL_DEPLOYMENT"]
UNDEPLOYED_STATUSES = ["SUCCESSFUL_UNDEPLOYMENT"]
FAILED_STATUSES = [
    "INVALID_DEPLOYMENT",
    "FAILED_INITIATING_DEPLOYMENT",
    "FAILED_DEPLOYMENT",
    "FAILED_UNDEPLOYMENT",
]


@click.command("stream", help="Deploy stream model.", cls=QwakCommand)
@click.option(
    "--build-id",
    help="Build ID, If not specified latest successful build will deployed.",
    type=str,
)
@click.option("--model-id", help="Model ID", type=str, required=True)
@click.option(
    "--environment-name",
    required=False,
    type=str,
    help="Environment to deploy on (if not specified uses your default environment",
    multiple=True,
)
@click.option(
    "--bootstrap-server",
    envvar="BOOTSTRAP_SERVERS",
    help="Kafka consumer/producer bootstrap server.",
    multiple=True,
    type=str,
)
@click.option(
    "--consumer-bootstrap-server",
    envvar="CONSUMER_BOOTSTRAP_SERVER",
    help="Kafka consumer bootstrap server.",
    multiple=True,
    type=str,
)
@click.option(
    "--consumer-topic",
    envvar="CONSUMER_TOPIC",
    help="Kafka consumer topic.",
    type=str,
)
@click.option(
    "--consumer-group",
    envvar="CONSUMER_GROUP",
    help="Kafka consumer group.",
    type=str,
)
@click.option(
    "--consumer-auto-offset-reset",
    envvar="CONSUMER_AUTO_OFFSET_RESET",
    type=click.Choice([i.value for i in AutoOffsetReset], case_sensitive=False),
    help="Kafka consumer auto offset reset (Latest/Earliest).",
)
@click.option(
    "--consumer-timeout",
    envvar="CONSUMER_TIMEOUT",
    help="Kafka consumer polling timeout. (The timeout should be in range of kafka admin configuration "
    "'group.min.session.timeout.ms' and 'group.max.session.timeout.ms'.)",
    type=int,
)
@click.option(
    "--consumer-max-batch-size",
    envvar="CONSUMER_MAX_BATCH_SIZE",
    help="The maximum number of records returned in a single call to poll().",
    type=int,
)
@click.option(
    "--consumer-max-poll-latency",
    envvar="CONSUMER_MAX_POLL_LATENCY",
    help="The maximum delay between invocations of poll() when using consumer group management.",
    type=float,
)
@click.option(
    "--producer-bootstrap-server",
    envvar="PRODUCER_BOOTSTRAP_SERVER",
    help="Kafka producer bootstrap server.",
    multiple=True,
    type=str,
)
@click.option(
    "--producer-topic",
    envvar="PRODUCER_TOPIC",
    help="Kafka producer topic.",
    type=str,
)
@click.option(
    "--producer-compression-type",
    envvar="PRODUCER_COMPRESSION_TYPE",
    type=click.Choice([i.value for i in CompressionTypes], case_sensitive=False),
    help="Kafka producer compression type.",
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
    "--local",
    is_flag=True,
    default=False,
    help="Run deployments locally.",
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
def stream(
    verbose: bool, from_file: str, out_conf: bool, local: bool, sync: bool, **kwargs
):
    set_qwak_logger_stdout_verbosity_level(verbose + 1)
    config: DeployConfig = config_handler(
        config=DeployConfig,
        from_file=from_file,
        out_conf=out_conf,
        sections=("stream",),
        **kwargs,
    )
    if not out_conf:
        deploy_executor = StreamDeployExecutor(config)
        client_deployment(deploy=deploy_executor, sync=sync)
