import click
from qwak.clients.feature_store import FeatureRegistryClient
from qwak.exceptions import QwakException

from qwak_sdk.inner.tools.cli_tools import QwakCommand
from qwak_sdk.tools.colors import Color


@click.command("pause", cls=QwakCommand, help="Pause a running feature set")
@click.argument("name")
def pause_feature_set(name, **kwargs):
    try:
        FeatureRegistryClient().pause_feature_set(feature_set_name=name)
    except Exception as e:
        print(f"{Color.RED} Failed to pause feature set {name} {Color.END}")
        raise QwakException(f"Failed to pause feature set {name}") from e

    print(f"{Color.GREEN}Successfully paused feature set {Color.YELLOW}{name}")
