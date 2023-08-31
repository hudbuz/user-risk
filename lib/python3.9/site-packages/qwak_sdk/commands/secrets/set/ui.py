import click

from qwak_sdk.commands.secrets.set._logic import execute_set_secret
from qwak_sdk.inner.tools.cli_tools import QwakCommand


@click.command("set", cls=QwakCommand)
@click.option("--name", metavar="TEXT", required=True, help="the secret name")
@click.option("--value", hide_input=True, confirmation_prompt=False, prompt=True)
def set_secret(name, value, **kwargs):
    print(f"Creating secret named '{name}' with value length of {len(value)}...")
    try:
        execute_set_secret(name, value)
        print("Created!")
    except Exception as e:
        print(f"Error setting secret. Error is {e}")
