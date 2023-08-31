from qwak.clients.workspace_manager import WorkspaceManagerClient

from qwak_sdk.commands.workspaces._logic.workspace_validations import (
    _validate_workspace_name,
    _verify_image_id,
    _verify_instance,
)
from qwak_sdk.commands.workspaces.config.workspace_config import WorkspaceConfig


def _create_workspace(config: WorkspaceConfig):
    """
    Creating a new workspace

    Args:
        config: The workspace configuration

    """
    print(f"Creating a new workspace with the name {config.workspace.name}")
    workspace_manager_client = WorkspaceManagerClient()
    _validate_workspace_name(config.workspace.name)
    _verify_image_id(config.workspace.image, workspace_manager_client)
    _verify_instance(config.workspace.instance)

    response = workspace_manager_client.create_workspace(
        config.workspace.name, config.workspace.image, config.workspace.instance
    )

    print(
        f"Workspace {config.workspace.name} was created successfully with id: {response.workspace_id}"
    )
