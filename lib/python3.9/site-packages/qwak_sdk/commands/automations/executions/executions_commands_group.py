import click

from qwak_sdk.commands.automations.executions.list.ui import list_executions

DELIMITER = "----------------------------------------"


@click.group(name="executions", help=" automation executions")
def executions_command_group():
    # Click commands group injection
    pass


executions_command_group.add_command(list_executions)
