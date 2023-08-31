import click

from qwak_sdk.commands.automations.delete._logic import delete_automation
from qwak_sdk.inner.tools.cli_tools import QwakCommand


@click.command(
    "delete",
    help="Delete an automation object",
    cls=QwakCommand,
)
@click.option(
    "--automation-id",
    type=str,
    metavar="ID",
    help="The automation id to delete",
)
def delete(automation_id: str, **kwargs):
    deleted = delete_automation(automation_id=automation_id)
    if deleted:
        print(f"Automation {automation_id} deleted successfully")
    else:
        print(f"Automation {automation_id} was not found")
