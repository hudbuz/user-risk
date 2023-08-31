import click
from qwak.exceptions import QwakException

from qwak_sdk.commands.models.runtime.update._logic import execute_runtime_update
from qwak_sdk.inner.tools.cli_tools import QwakCommand


@click.command("update", cls=QwakCommand)
@click.option("-l", "--log-level", required=True, help="Log level to set.")
@click.option("-m", "--model-id", required=True, help="Model named ID")
def runtime_update(model_id, log_level, **kwargs):
    try:
        execute_runtime_update(model_id, log_level=log_level)
    except Exception as e:
        raise QwakException(f'Failed to update runtime configurations. Error is "{e}"')
