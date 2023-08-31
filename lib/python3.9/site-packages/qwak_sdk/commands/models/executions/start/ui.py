import click
from qwak.clients.batch_job_management.executions_config import (
    GPU_TYPE_MAP,
    INPUT_FORMATTERS_MAP,
    OUTPUT_FORMATTERS_MAP,
    ExecutionConfig,
)
from qwak.exceptions import QwakException
from qwak.inner.const import QwakConstants
from qwak.tools.logger.logger import get_qwak_logger
from tabulate import tabulate

from qwak_sdk.commands.models.executions.start._logic import (
    execute_in_local_mode,
    execute_start_execution,
)
from qwak_sdk.inner.tools.cli_tools import DeprecatedOption, QwakCommand
from qwak_sdk.inner.tools.config_handler import config_handler

logger = get_qwak_logger()


@click.command(
    name="start", help="Start an execution job on a deployed model.", cls=QwakCommand
)
@click.option("--model-id", required=True, help="Model ID", type=str)
@click.option(
    "--build-id",
    type=str,
    help="Explicit Build ID to invoke the execution against. If empty, the deployed build ID will be applied",
)
@click.option(
    "--bucket",
    type=str,
    help="The bucket to read and write the data to/from.",
)
@click.option(
    "--source-bucket",
    type=str,
    help="The source bucket to read the data from.",
)
@click.option(
    "--source-folder",
    required=True,
    type=str,
    help="The source folder to read the data from.",
)
@click.option(
    "--destination-bucket",
    type=str,
    help="The destination bucket to store results on.",
)
@click.option(
    "--destination-folder",
    type=str,
    required=True,
    help="The destination folder to store results on.",
)
@click.option(
    "--input-file-type",
    type=click.Choice(
        [fmt for fmt in INPUT_FORMATTERS_MAP.keys()], case_sensitive=False
    ),
    help="The input file type to read.",
)
@click.option(
    "--output-file-type",
    type=click.Choice(
        [fmt for fmt in OUTPUT_FORMATTERS_MAP.keys()], case_sensitive=False
    ),
    help="The output file type to to receive.",
)
@click.option(
    "--access-token-name",
    type=str,
    required=False,
    help="The access token secret name (should be created before by using `qwak secret`).",
)
@click.option(
    "--access-secret-name",
    type=str,
    required=False,
    help="The access secret secret name (should be created before by using `qwak secret`).",
)
@click.option(
    "--job-timeout",
    type=int,
    default=0,
    help="Total inference job timeout in seconds, (e.g. 2000). Default is unlimited.",
)
@click.option(
    "--file-timeout",
    type=int,
    default=0,
    help="A single file inference timeout in seconds, (e.g. 2000). Default is unlimited.",
)
@click.option(
    "--pods",
    "--replicas",
    type=int,
    default=0,
    help="Number of replicas to deploy. (concurrent files processing). Default is as deployed.",
    deprecated=["--pods"],
    preferred="--replicas",
    cls=DeprecatedOption,
)
@click.option(
    "--cpus",
    type=float,
    default=0,
    help="Number of CPU cores, can be fraction of a core "
    "(e.g. 0.4). Default is as deployed.",
)
@click.option(
    "--memory",
    type=int,
    default=0,
    help="Memory amount in Mib (e.g. 256). Default is as deployed.",
)
@click.option(
    "--gpu-type",
    type=click.Choice([fmt for fmt in GPU_TYPE_MAP.keys()], case_sensitive=False),
    help=f"Type of GPU to use on the deployment ({', '.join([x for x in QwakConstants.GPU_TYPES])}).",
)
@click.option(
    "--gpu-amount",
    type=int,
    help="Amount of GPU's to use on the in the batch inference.",
)
@click.option(
    "-f",
    "--from-file",
    help="Build by run_config file, Command arguments will overwrite any run_config.",
    required=False,
    type=click.Path(exists=True, resolve_path=True, dir_okay=False),
)
@click.option(
    "--out-conf",
    help="Extract deploy conf from command arguments, the command will not run it wil only output valid yaml "
    "structure",
    default=False,
    is_flag=True,
)
@click.option(
    "-P",
    "--param-list",
    required=False,
    metavar="NAME=VALUE",
    multiple=True,
    help="A parameter for the batch execution, of the form -P name=value",
)
@click.option(
    "--instance",
    required=False,
    type=str,
    help="The instance size to run batch on - 'small', 'medium', 'large', etc...",
    default=None,
)
def execution_start(from_file: str, out_conf: bool, **kwargs):
    if kwargs["source_folder"].startswith("file:") and kwargs[
        "destination_folder"
    ].startswith("file:"):
        execute_in_local_mode(kwargs)
        return
    else:
        access_token_name = kwargs.get("access_token_name", None)
        access_secret_name = kwargs.get("access_secret_name", None)
        if access_token_name is None or access_secret_name is None:
            raise QwakException(
                "Both access-token-name and access-secret-name must be provided"
            )

    config: ExecutionConfig = config_handler(
        config=ExecutionConfig,
        from_file=from_file,
        out_conf=out_conf,
        sections=("execution", "resources"),
        **kwargs,
    )
    if not out_conf:
        try:
            execution_id, success, fail_message = execute_start_execution(config)
            if success:
                logger.info(
                    tabulate(
                        tabular_data=[
                            ["Model ID", config.execution.model_id],
                            ["Execution ID", execution_id],
                        ],
                        tablefmt="fancy_grid",
                    )
                )
            else:
                raise QwakException(fail_message + f" {execution_id}")

            logger.info(
                "Execution initiated successfully, Use --status to get execution current status."
            )
        except Exception as e:
            logger.error(f"Execution failed, Error: {e}")
            raise
