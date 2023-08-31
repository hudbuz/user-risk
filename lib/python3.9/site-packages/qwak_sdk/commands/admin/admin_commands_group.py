import click

from qwak_sdk.commands.admin.apikeys.api_keys_commands_group import (
    api_keys_commands_group,
)


@click.group(
    name="admin",
    help="Commands for admin operations",
)
def admin_commands_group():
    # Click commands group injection
    pass


admin_commands_group.add_command(api_keys_commands_group)
