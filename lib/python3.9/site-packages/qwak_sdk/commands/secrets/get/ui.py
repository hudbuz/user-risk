import click
from qwak.exceptions import QwakNotFoundException

from qwak_sdk.commands.secrets.get._logic import execute_get_secret
from qwak_sdk.inner.tools.cli_tools import QwakCommand


@click.command("get", cls=QwakCommand)
@click.option("--name", metavar="TEXT", required=True, help="the secret name")
def get_secret(name, **kwargs):
    try:
        value = execute_get_secret(name)
        print(value)
    except QwakNotFoundException:
        print(f"Secret '{name}' does not exists")
    except Exception as e:
        print(f"Error getting secret. Error is {e}")
