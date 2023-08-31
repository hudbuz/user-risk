import click

from qwak_sdk.commands.auto_scalling.attach._logic import attach_autoscaling
from qwak_sdk.exceptions import QwakCommandException
from qwak_sdk.inner.tools.cli_tools import QwakCommand


@click.command("attach", cls=QwakCommand)
@click.option("-m", "--model-id", required=False, help="Model named ID")
@click.option("-v", "--variation-name", required=False, help="Model variation name")
@click.option("-f", "--file", required=True, type=click.Path(exists=True))
def attach(model_id: str, variation_name: str, file: str, **kwargs):
    click.echo("Attaching autoscaling configuration")
    try:
        response = attach_autoscaling(model_id, variation_name, file)
        click.echo(
            f"Successfully configured autoscaling. model: {response['model_id']}, variation: {response['variation_name']}"
        )
    except QwakCommandException as e:
        click.echo(f"Failed to attach autoscaling, Error: {e}")
        exit(1)
