from typing import Optional

from qwak.clients.workspace_manager import WorkspaceManagerClient
from qwak.exceptions import QwakException


def get_workspace_id_by_name(
    workspace_name: str, workspace_manager_client: Optional[WorkspaceManagerClient]
) -> str:
    """
    Args:
        workspace_name: The name of the workspace
    Returns:
        The id of the workspace
    """
    client = (
        workspace_manager_client
        if workspace_manager_client
        else WorkspaceManagerClient()
    )

    response = client.get_workspaces()

    for workspace in response.workspaces:
        if workspace.workspace_name == workspace_name:
            return workspace.workspace_id

    raise QwakException(f"Failed to find workspace with name {workspace_name}")
