import json

import click
from _qwak_proto.qwak.builds.builds_pb2 import BuildStatus
from qwak.tools.logger.logger import get_qwak_logger

from qwak_sdk.commands.models.builds.status._logic import execute_get_build_status
from qwak_sdk.inner.tools.cli_tools import QwakCommand

logger = get_qwak_logger()


@click.command("status", cls=QwakCommand)
@click.argument("build_id")
@click.option(
    "--format",
    default="text",
    show_default=True,
    type=click.Choice(["text", "json"], case_sensitive=True),
    metavar="FORMAT",
    required=False,
    help="The formatting style for commands output (choose from text, json)",
)
def get_build_status(build_id, format, **kwargs):
    if format == "text":
        logger.info(f"Getting build status for build id [{build_id}]")
    build_status = execute_get_build_status(build_id)
    if format == "text":
        logger.info(f"Build status: {BuildStatus.Name(build_status)}")
    elif format == "json":
        print(
            json.dumps(
                {
                    "build_id": build_id,
                    "build_status": BuildStatus.Name(build_status),
                }
            )
        )
    return BuildStatus.Name(build_status)
