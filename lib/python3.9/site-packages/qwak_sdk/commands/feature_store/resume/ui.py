import click
from qwak.clients.feature_store import FeatureRegistryClient
from qwak.exceptions import QwakException

from qwak_sdk.inner.tools.cli_tools import QwakCommand
from qwak_sdk.tools.colors import Color


@click.command("resume", cls=QwakCommand, help="Resume a paused feature set")
@click.argument("name")
def resume_feature_set(name, **kwargs):
    try:
        FeatureRegistryClient().resume_feature_set(feature_set_name=name)
    except Exception as e:
        print(f"{Color.RED} Failed to resume feature set {name} {Color.END}")
        raise QwakException(f"Failed to resume feature set {name}") from e

    print(f"{Color.GREEN}Successfully resume feature set {Color.YELLOW}{name}")
