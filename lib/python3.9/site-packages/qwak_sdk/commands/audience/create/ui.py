from pathlib import Path

import click

from qwak_sdk.commands.audience.create.logic import create_audience
from qwak_sdk.exceptions import QwakCommandException, QwakResourceNotFound
from qwak_sdk.inner.tools.cli_tools import QwakCommand


@click.command("create", cls=QwakCommand)
@click.option("--name", type=str)
@click.option("--description", type=str)
@click.option("-f", "--file", type=click.Path(exists=True))
def create_audience_command(name: str, description: str, file: Path, **kwargs):
    click.echo("Creating audience")
    try:
        audience_id = create_audience(name=name, description=description, file=file)
        click.echo(f"Audience ID {audience_id} created successfully")
    except (QwakCommandException, QwakResourceNotFound) as e:
        click.echo(f"Failed to create audience, Error: {e}")
        exit(1)
