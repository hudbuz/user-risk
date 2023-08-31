import click

from qwak_sdk.commands.alerts.delete._logic import execute_delete_channel
from qwak_sdk.inner.tools.cli_tools import QwakCommand


@click.command("delete", cls=QwakCommand, help="Delete a Channel by name")
@click.argument("name")
def delete_channel(name, **kwargs):
    execute_delete_channel(name)
