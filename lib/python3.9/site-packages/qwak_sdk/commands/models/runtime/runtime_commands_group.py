import click

from qwak_sdk.commands.models.runtime.logs.ui import runtime_logs
from qwak_sdk.commands.models.runtime.traffic_update.ui import runtime_traffic_update
from qwak_sdk.commands.models.runtime.update.ui import runtime_update


@click.group(
    name="runtime",
    help="Runtime configurations for deployed models",
)
def runtime_commands_group():
    # Click commands group injection
    pass


runtime_commands_group.add_command(runtime_logs)
runtime_commands_group.add_command(runtime_traffic_update)
runtime_commands_group.add_command(runtime_update)
