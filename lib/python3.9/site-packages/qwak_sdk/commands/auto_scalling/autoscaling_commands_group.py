import click

from qwak_sdk.commands.auto_scalling.attach.ui import attach


@click.group(
    name="autoscaling",
    help="Manage autoscaling",
)
def autoscaling_commands_group():
    # Click commands group injection
    pass


autoscaling_commands_group.add_command(attach)
