import click

from qwak_sdk.commands.projects.create.ui import create_project
from qwak_sdk.commands.projects.delete.ui import delete_project
from qwak_sdk.commands.projects.list.ui import projects_list


@click.group(
    name="projects",
    help="Commands for interacting with Qwak based project",
)
def projects_command_group():
    # Intentionally left empty since this is how we inject this CLI to the general CLI group
    pass


projects_command_group.add_command(create_project)
projects_command_group.add_command(projects_list)
projects_command_group.add_command(delete_project)
