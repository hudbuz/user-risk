import click

from qwak_sdk.commands.models.builds.cancel.ui import cancel_build
from qwak_sdk.commands.models.builds.logs.ui import build_logs
from qwak_sdk.commands.models.builds.status.ui import get_build_status


@click.group(name="builds", help="Ongoing builds")
def builds_commands_group():
    # Click commands group injection
    pass


builds_commands_group.add_command(cancel_build)
builds_commands_group.add_command(get_build_status)
builds_commands_group.add_command(build_logs)
