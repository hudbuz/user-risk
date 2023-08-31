import click
import grpc

from qwak_sdk.commands.secrets.delete._logic import execute_delete_secret
from qwak_sdk.inner.tools.cli_tools import QwakCommand


@click.command("delete", cls=QwakCommand)
@click.option("--name", metavar="TEXT", required=True, help="The secret name")
def delete_secret(name, **kwargs):
    print(f"Deleting secret named '{name}'")
    try:
        execute_delete_secret(name)
        print(f"Secret '{name}' has been deleted")
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.NOT_FOUND:
            print(f"Secret '{name}' does not exist")
        else:
            print(f"Error deleting secret. Error is {e}")
    except Exception as e:
        print(f"Error deleting secret. Error is {e}")
