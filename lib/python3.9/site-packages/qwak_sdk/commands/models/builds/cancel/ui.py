import click
from qwak.tools.logger.logger import get_qwak_logger

from qwak_sdk.commands.models.builds.cancel._logic import execute_cancel_build
from qwak_sdk.inner.tools.cli_tools import QwakCommand

logger = get_qwak_logger()


@click.command("cancel", cls=QwakCommand)
@click.argument("build_id")
def cancel_build(build_id, **kwargs):
    logger.info(f"Attempting to cancel remote build with build id [{build_id}]")
    execute_cancel_build(build_id=build_id)
    logger.info("Successfully canceled remote build")
