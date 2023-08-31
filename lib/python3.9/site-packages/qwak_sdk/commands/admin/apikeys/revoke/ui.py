import click
from qwak.tools.logger.logger import get_qwak_logger

from qwak_sdk.commands.admin.apikeys.revoke._logic import execute_revoke_apikey
from qwak_sdk.inner.tools.cli_tools import QwakCommand

logger = get_qwak_logger()


@click.command("revoke", help="api keys", cls=QwakCommand)
@click.option(
    "--environment-name",
    help="Environment name to generate api key for",
    required=True,
)
@click.option(
    "--user-id",
    help="User id to generate api key for",
    required=True,
)
def revoke_apikey(user_id, environment_name, **kwargs):
    logger.info(
        f"Revoking api key for user {user_id} on environment {environment_name}..."
    )
    try:
        if execute_revoke_apikey(user_id=user_id, environment_name=environment_name):
            logger.info("Successfully revoked api key")
        else:
            logger.info("Failed to revoke api key")
    except Exception as e:
        logger.error(e.message)
