import click
from qwak.clients.feature_store import FeatureRegistryClient
from qwak.exceptions import QwakException

from qwak_sdk.inner.tools.cli_tools import QwakCommand
from qwak_sdk.tools.colors import Color


@click.command(
    "trigger", cls=QwakCommand, help="Trigger a batch feature set job ingestion job"
)
@click.argument("name")
def trigger_feature_set(name, **kwargs):
    """
    Trigger a batch feature set ingestion job

    Args:
        name: feature set name that will be triggered
    """
    try:
        FeatureRegistryClient().run_feature_set(feature_set_name=name)
    except Exception as e:
        print(
            f"{Color.RED} Failed to trigger a batch feature set ingestion for feature set {name} {Color.END}"
        )
        raise QwakException(
            f"Failed to trigger a batch feature set ingestion for feature set {name}"
        ) from e

    print(
        f"{Color.GREEN}Successfully triggered a batch feature set ingestion for feature set {Color.YELLOW}{name}"
    )
