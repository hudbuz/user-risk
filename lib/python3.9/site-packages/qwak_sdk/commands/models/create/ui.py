import click

from qwak_sdk.commands.models.create._logic import execute_model_create
from qwak_sdk.commands.ui_tools import output_as_json
from qwak_sdk.inner.tools.cli_tools import QwakCommand
from qwak_sdk.tools.colors import Color


@click.command("create", cls=QwakCommand)
@click.argument("name", metavar="name", required=True)
@click.option("--project", metavar="NAME", required=False, help="Project name")
@click.option("--project-id", metavar="ID", required=False, help="Project id")
@click.option(
    "--description",
    metavar="DESCRIPTION",
    required=False,
    help="Model description",
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
def model_create(name, description, project, project_id, format, **kwargs):
    try:
        response = execute_model_create(name, description, project, project_id)
        if format == "json":
            output_as_json(response)
        else:
            print(f"Model created\nmodel id : {response.model_id}")
    except Exception as e:
        print(f"{Color.RED}Error creating model: {e}{Color.RED}")
