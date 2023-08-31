import click
from qwak.clients.feature_store import FeatureRegistryClient

from qwak_sdk.commands.feature_store.delete._logic import (
    DATA_SOURCE,
    ENTITY,
    FEATURE_SET,
    _inner_delete,
)
from qwak_sdk.inner.tools.cli_tools import QwakCommand


@click.command(
    "delete",
    cls=QwakCommand,
    help="Delete by name a feature store object - a feature set, entity, or data source",
)
@click.option(
    "--feature-set",
    "object",
    flag_value=FEATURE_SET,
    default=True,
    help="Delete a feature set [default option]",
)
@click.option(
    "--data-source", "object", flag_value=DATA_SOURCE, help="Delete a data source"
)
@click.option("--entity", "object", flag_value=ENTITY, help="Delete an entity")
@click.argument("name")
def delete_fs_object(object, name, **kwargs):
    """
    Delete Feature Store object

    Args:
        object: type of object to delete
        name: name of the object to delete
    """

    registry_client = FeatureRegistryClient()
    _inner_delete(name, object, registry_client)
