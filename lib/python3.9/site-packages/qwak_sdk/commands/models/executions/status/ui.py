import click
from qwak.exceptions import QwakException
from qwak.tools.logger.logger import get_qwak_logger

from qwak_sdk.commands.models.executions.status._logic import execute_execution_status
from qwak_sdk.inner.tools.cli_tools import QwakCommand

logger = get_qwak_logger()


@click.command(name="status", help="Get the status of an execution.", cls=QwakCommand)
@click.option(
    "--execution-id",
    type=str,
    help="The execution id.",
)
def execution_status(execution_id: str, **kwargs):
    try:
        job_status, success, fail_message = execute_execution_status(execution_id)
        if not success:
            raise QwakException(fail_message)

        logger.info(f"Status is: {job_status}.")

    except Exception as e:
        logger.error(f"Failed to cancel execution, Error: {e}")
        raise
