import click
from google.protobuf.json_format import MessageToJson

from qwak_sdk.commands.projects.delete._logic import execute as execute_delete
from qwak_sdk.inner.tools.cli_tools import QwakCommand


@click.command("delete", cls=QwakCommand)
@click.option("--project-id", metavar="NAME", required=True, help="Project name")
@click.option(
    "--format",
    default="text",
    show_default=True,
    type=click.Choice(["text", "json"], case_sensitive=True),
    metavar="FORMAT",
    required=False,
    help="The formatting style for commands output (choose from text, json)",
)
def delete_project(project_id, format, **kwargs):
    response = execute_delete(project_id)
    if format == "json":
        print(MessageToJson(response))
    else:
        print(f"Project deleted\nproject id : {project_id}")
