import click

from qwak_sdk.commands.projects.create._logic import execute as execute_create
from qwak_sdk.commands.ui_tools import output_as_json
from qwak_sdk.inner.tools.cli_tools import QwakCommand


@click.command("create", cls=QwakCommand)
@click.argument("name", metavar="name", required=True)
@click.option(
    "--description",
    metavar="DESCRIPTION",
    required=False,
    help="Project description",
)
@click.option(
    "--format",
    default="text",
    show_default=True,
    type=click.Choice(["text", "json"], case_sensitive=True),
    metavar="FORMAT",
    required=False,
    help="The formatting style for commands output (choose from text, json)",
)
def create_project(name, description, format, *args, **kwargs):
    response = execute_create(name, description)
    if format == "json":
        output_as_json(response)
    else:
        print(f"Project created\nproject id : {response.project.project_id}")
