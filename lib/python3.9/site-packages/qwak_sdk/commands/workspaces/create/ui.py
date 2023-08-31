from typing import Optional

import click

from qwak_sdk.commands.workspaces.config.workspace_config import WorkspaceConfig
from qwak_sdk.commands.workspaces.create._logic import _create_workspace
from qwak_sdk.inner.tools.cli_tools import QwakCommand
from qwak_sdk.inner.tools.config_handler import config_handler


@click.command("create", cls=QwakCommand)
@click.option(
    "--name",
    required=False,
    metavar="NAME",
    help="The name of the requested workspace",
)
@click.option(
    "--instance",
    required=False,
    metavar="INSTANCE",
    help="The instance id of the requested template",
)
@click.option(
    "--image",
    required=False,
    metavar="IMAGE",
    help="The image for the requested workspace",
)
@click.option(
    "--out-conf",
    help="Extract workspace conf from commands arguments, the commands will not run it wil only output valid yaml ",
    default=False,
    is_flag=True,
)
@click.option(
    "--from-file",
    required=False,
    help="Create by run_config file, Command arguments will overwrite any run_config.",
)
def create_workspace(from_file: Optional[str], out_conf: bool, **kwargs):
    config: WorkspaceConfig = config_handler(
        config=WorkspaceConfig,
        from_file=from_file,
        **kwargs,
        out_conf=out_conf,
    )
    _create_workspace(config)
