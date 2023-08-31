import click

from qwak_sdk.commands.audience.audience_api_dump import (
    audience_entries_to_pretty_string,
)
from qwak_sdk.commands.audience.list.logic import list_audience
from qwak_sdk.exceptions import QwakCommandException, QwakResourceNotFound
from qwak_sdk.inner.tools.cli_tools import QwakCommand


@click.command("list", cls=QwakCommand)
def list_audience_command(**kwargs):
    click.echo("Getting audiences")
    try:
        audience_entries = list_audience()
        click.echo(audience_entries_to_pretty_string(audience_entries=audience_entries))
    except (QwakCommandException, QwakResourceNotFound) as e:
        click.echo(f"Failed to list audiences, Error: {e}")
        exit(1)
