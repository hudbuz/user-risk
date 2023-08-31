from datetime import datetime

import click

from qwak_sdk.commands.models.list._logic import execute_models_list
from qwak_sdk.commands.ui_tools import output_as_json, output_as_table
from qwak_sdk.inner.tools.cli_tools import QwakCommand


def parse_model(model):
    return [
        model.model_id,
        model.display_name,
        datetime.fromtimestamp(
            model.created_at.seconds + model.created_at.nanos / 1e9
        ).strftime("%A, %B %d, %Y %I:%M:%S"),
        datetime.fromtimestamp(
            model.last_modified_at.seconds + model.last_modified_at.nanos / 1e9
        ).strftime("%A, %B %d, %Y %I:%M:%S"),
    ]


@click.command("list", cls=QwakCommand)
@click.option("--project-id", metavar="NAME", required=True, help="Project id")
@click.option(
    "--format",
    default="text",
    show_default=True,
    type=click.Choice(["text", "json"], case_sensitive=True),
    metavar="FORMAT",
    required=False,
    help="The formatting style for commands output (choose from text, json)",
)
def model_list(project_id, format, **kwargs):
    model_list_result = execute_models_list(project_id)
    columns = ["Model id", "Display name", "Creation date", "Last updated"]
    if format == "json":
        output_as_json(model_list_result)
    elif format == "text":
        output_as_table(model_list_result.project.models, parse_model, headers=columns)
