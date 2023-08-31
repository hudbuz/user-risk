import click

from qwak_sdk.commands.alerts.list._logic import execute_list_channels
from qwak_sdk.inner.tools.cli_tools import QwakCommand


@click.command("list", cls=QwakCommand, help="List all alerts system channels.")
def list_channels(**kwargs):
    execute_list_channels()
