from datetime import datetime

import click

from qwak_sdk.commands.projects.list._logic import execute as execute_list
from qwak_sdk.commands.ui_tools import output_as_json, output_as_table
from qwak_sdk.inner.tools.cli_tools import QwakCommand


def parse_project(project):
    return [
        project.project_id,
        project.project_name,
        datetime.fromtimestamp(
            project.created_at.seconds + project.created_at.nanos / 1e9
        ).strftime("%A, %B %d, %Y %I:%M:%S"),
        datetime.fromtimestamp(
            project.last_modified_at.seconds + project.last_modified_at.nanos / 1e9
        ).strftime("%A, %B %d, %Y %I:%M:%S"),
    ]


@click.command("list", cls=QwakCommand)
@click.option(
    "--format",
    default="text",
    show_default=True,
    type=click.Choice(["text", "json"], case_sensitive=True),
    metavar="FORMAT",
    required=False,
    help="The formatting style for commands output (choose from text, json)",
)
def projects_list(format, **kwargs):
    projects_data = execute_list()
    if format == "json":
        output_as_json(projects_data)
    elif format == "text":
        columns = [
            "Project id",
            "Project name",
            "Creation date",
            "Last updated",
            "Models count",
        ]
        output_as_table(projects_data.projects, parse_project, columns)
