from pathlib import Path

import click

from qwak_sdk.commands.alerts.register._logic import execute_register_channel
from qwak_sdk.inner.tools.cli_tools import QwakCommand


@click.command(
    "register",
    cls=QwakCommand,
    help="Register all alerts system objects under the given path.",
)
@click.option(
    "--path",
    "-p",
    type=click.Path(exists=True),
    metavar="PATH",
    help="Directory / module where qwak alerts system objects are stored",
)
@click.option(
    "--force",
    "-f",
    default=False,
    is_flag=True,
    metavar="FLAG",
    help="Force register all found qwak alerts system objects",
)
def register_channel(path: Path, force: bool, **kwargs):
    execute_register_channel(path, force)
