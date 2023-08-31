import os
from pathlib import Path

import click

from qwak_sdk.commands.automations.register._logic import register_automations
from qwak_sdk.inner.file_registry import list_qwak_python_files
from qwak_sdk.inner.tools.cli_tools import QwakCommand
from qwak_sdk.tools.utils import qwak_spinner


@click.command(
    "register",
    help="Register all automations object under the given path. Registered "
    "automations will be visible on the Qwak management platform after registration",
    cls=QwakCommand,
)
@click.option(
    "--path",
    "-p",
    type=click.Path(exists=True),
    metavar="PATH",
    help="Directory / module where qwak automations objects are stored",
)
@click.option(
    "--force",
    "-f",
    default=False,
    is_flag=True,
    metavar="FLAG",
    help="Force register all found qwak automations Store objects",
)
def register(path: Path, force: bool, **kwargs):
    path = Path(path) if path else Path.cwd()
    if path.is_file():
        qwak_python_files = [(str(path), os.path.abspath(path))]
    elif Path.is_dir(path):
        with qwak_spinner(
            begin_text="Recursively looking for python files in input dir",
            print_callback=print,
        ) as sp:
            qwak_python_files = list_qwak_python_files(path, sp)

    register_automations(qwak_python_files, force)
