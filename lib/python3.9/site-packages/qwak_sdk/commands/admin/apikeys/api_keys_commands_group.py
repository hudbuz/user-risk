import click
from qwak.tools.logger.logger import get_qwak_logger

from qwak_sdk.commands.admin.apikeys.generate.ui import generate_apikey
from qwak_sdk.commands.admin.apikeys.revoke.ui import revoke_apikey

logger = get_qwak_logger()


@click.group(name="api-keys", help="api keys")
def api_keys_commands_group():
    # Click commands group injection
    pass


api_keys_commands_group.add_command(generate_apikey)
api_keys_commands_group.add_command(revoke_apikey)
