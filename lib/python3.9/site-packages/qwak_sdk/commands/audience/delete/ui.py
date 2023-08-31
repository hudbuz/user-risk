import click

from qwak_sdk.commands.audience.delete.logic import delete_audience
from qwak_sdk.exceptions import QwakCommandException, QwakResourceNotFound
from qwak_sdk.inner.tools.cli_tools import QwakCommand


@click.command("delete", cls=QwakCommand)
@click.option("--audience-id", type=str)
def delete_audience_command(audience_id: str, **kwargs):
    click.echo(f"Deleting audience ID {audience_id}")
    try:
        delete_audience(audience_id)
        click.echo(f"Audience ID {audience_id} deleted successfully")
    except (QwakCommandException, QwakResourceNotFound) as e:
        click.echo(f"Failed to delete audience, Error: {e}")
        exit(1)
