from typing import List

import click
from qwak.automations import Automation

from qwak_sdk.commands.automations.list._logic import execute_list_automations
from qwak_sdk.inner.tools.cli_tools import QwakCommand

DELIMITER = "----------------------------------------"


@click.command(
    "list",
    help="List all automations",
    cls=QwakCommand,
)
def list_automations(**kwargs):
    automations_list: List[Automation] = execute_list_automations()
    for automation in automations_list:
        print(automation)
        print(DELIMITER)
