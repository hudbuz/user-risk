import click

from qwak_sdk.commands.feature_store.delete.ui import delete_fs_object
from qwak_sdk.commands.feature_store.list.ui import list_feature_sets
from qwak_sdk.commands.feature_store.pause.ui import pause_feature_set
from qwak_sdk.commands.feature_store.register.ui import register_fs_objects
from qwak_sdk.commands.feature_store.resume.ui import resume_feature_set
from qwak_sdk.commands.feature_store.trigger.ui import trigger_feature_set


@click.group(
    name="features",
    help="Commands for interacting with the Qwak Feature Store",
)
def feature_store_commands_group():
    # Click commands group injection
    pass


feature_store_commands_group.add_command(delete_fs_object)
feature_store_commands_group.add_command(list_feature_sets)
feature_store_commands_group.add_command(pause_feature_set)
feature_store_commands_group.add_command(resume_feature_set)
feature_store_commands_group.add_command(trigger_feature_set)
feature_store_commands_group.add_command(register_fs_objects)
