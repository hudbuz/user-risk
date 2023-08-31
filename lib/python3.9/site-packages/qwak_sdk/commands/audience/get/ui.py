import click

from qwak_sdk.commands.audience.audience_api_dump import (
    audience_to_json,
    audience_to_pretty_string,
)
from qwak_sdk.commands.audience.get.logic import get_audience
from qwak_sdk.exceptions import QwakCommandException, QwakResourceNotFound
from qwak_sdk.inner.tools.cli_tools import QwakCommand


@click.command("get", cls=QwakCommand)
@click.option("--audience-id", type=str)
@click.option("--json-format", is_flag=True, type=bool)
def get_audience_command(audience_id: str, json_format: bool, **kwargs):
    click.echo(f"Getting audience {audience_id}")
    try:
        audience = get_audience(audience_id)
        if json_format:
            click.echo(audience_to_json(audience))
        else:
            click.echo(audience_to_pretty_string(audience_id, audience))
    except (QwakCommandException, QwakResourceNotFound) as e:
        click.echo(f"Failed to get audience, Error: {e}")
        exit(1)
