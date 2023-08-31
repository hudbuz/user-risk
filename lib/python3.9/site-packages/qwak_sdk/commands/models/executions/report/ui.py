import click
from qwak.exceptions import QwakException
from qwak.tools.logger.logger import get_qwak_logger

from qwak_sdk.commands.models.executions.report._logic import execute_execution_report
from qwak_sdk.inner.tools.cli_tools import QwakCommand
from qwak_sdk.tools.colors import Color

logger = get_qwak_logger()


@click.command(name="report", help="Get the report of an execution.", cls=QwakCommand)
@click.option(
    "--execution-id",
    type=str,
    help="The execution id.",
)
def execution_report(execution_id: str, **kwargs):
    try:
        report_messages, model_logs, success, fail_message = execute_execution_report(
            execution_id
        )
        if not success:
            raise QwakException(fail_message)

        colored_report_lines = [
            f"{Color.BLUE}{message}{Color.END}" for message in report_messages
        ]

        colored_log_lines = [
            f"{Color.BLUE}{message}{Color.END}" for message in model_logs
        ]

        print(f"{Color.WHITE}Execution Report{Color.END}")
        print(f"{Color.WHITE}================{Color.END}")
        print("\n".join(colored_report_lines), end="\n")
        print(f"{Color.WHITE}Execution Logs{Color.END}")
        print(f"{Color.WHITE}=============={Color.END}")
        print("\n".join(colored_log_lines), end="\n")

    except Exception as e:
        logger.error(f"Failed to get execution report, Error: {e}")
        raise
