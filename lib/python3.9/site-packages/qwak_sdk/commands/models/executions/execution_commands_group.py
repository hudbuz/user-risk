import click
from qwak.tools.logger.logger import get_qwak_logger

from qwak_sdk.commands.models.executions.cancel.ui import execution_cancel
from qwak_sdk.commands.models.executions.report.ui import execution_report
from qwak_sdk.commands.models.executions.start.ui import execution_start
from qwak_sdk.commands.models.executions.status.ui import execution_status

logger = get_qwak_logger()


@click.group(
    name="execution",
    help="Executions of a batch job on deployed model.",
)
def execution_commands_group():
    # Click commands group injection
    pass


execution_commands_group.add_command(execution_cancel)
execution_commands_group.add_command(execution_report)
execution_commands_group.add_command(execution_start)
execution_commands_group.add_command(execution_status)
