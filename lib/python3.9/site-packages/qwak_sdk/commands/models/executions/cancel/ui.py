import click
from qwak.exceptions import QwakException
from qwak.tools.logger.logger import get_qwak_logger

from qwak_sdk.commands.models.executions.cancel._logic import execute_execution_cancel
from qwak_sdk.inner.tools.cli_tools import QwakCommand

logger = get_qwak_logger()


@click.command(name="cancel", help="Cancel a running execution.", cls=QwakCommand)
@click.option(
    "--execution-id",
    type=str,
    help="The execution id.",
)
def execution_cancel(execution_id: str, **kwargs):
    try:
        success, fail_message = execute_execution_cancel(execution_id)
        if not success:
            raise QwakException(fail_message)

        logger.info(f"Execution {execution_id} cancelled successfully.")

    except Exception as e:
        logger.error(f"Failed to cancel execution, Error: {e}")
        raise
