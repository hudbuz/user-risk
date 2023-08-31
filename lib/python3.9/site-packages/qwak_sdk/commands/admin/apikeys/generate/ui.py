import click
from qwak.clients.administration.self_service.client import APIKEY_ALREADY_EXISTS
from qwak.tools.logger.logger import get_qwak_logger

from qwak_sdk.commands.admin.apikeys.generate._logic import execute_generate_apikey
from qwak_sdk.inner.tools.cli_tools import QwakCommand

logger = get_qwak_logger()


@click.command("generate", help="api keys", cls=QwakCommand)
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
@click.option(
    "--force",
    help="If override existing one. (default false)",
    default=False,
    is_flag=True,
)
def generate_apikey(user_id, environment_name, force, **kwargs):
    logger.info(
        f"Generating api key for user {user_id} on environment {environment_name}..."
    )
    try:
        api_key = execute_generate_apikey(
            user_id=user_id, environment_name=environment_name, force=force
        )
        logger.info("generated api key:")
        logger.info(api_key)
    except Exception as e:
        if e.message == APIKEY_ALREADY_EXISTS:
            logger.error(
                "Failed to generate api key - "
                "the api key already exists please add --force if you want to override it."
            )
        else:
            logger.error(e)
