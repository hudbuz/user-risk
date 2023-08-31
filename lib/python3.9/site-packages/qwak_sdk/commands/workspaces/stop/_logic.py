from typing import Optional

from qwak.clients.workspace_manager import WorkspaceManagerClient

from qwak_sdk.commands.workspaces._logic.tools import get_workspace_id_by_name
from qwak_sdk.commands.workspaces._logic.workspace_validations import (
    validate_id_or_name,
)


def _stop_workspace(workspace_id: Optional[str], workspace_name: Optional[str]):
    """
    Undeploy an existing workspace
    Args:
    workspace_id: The id of the workspace to deploy
    """
    print(f"Undeploy an existing workspace with id {workspace_id}")
    validate_id_or_name(workspace_id=workspace_id, workspace_name=workspace_name)
    workspace_manager_client = WorkspaceManagerClient()
    if workspace_id:
        workspace_manager_client.undeploy_workspace(workspace_id=workspace_id)
        print(f"Workspace {workspace_id} was undeployed successfully")
    if workspace_name:
        workspace_id = get_workspace_id_by_name(
            workspace_name=workspace_name,
            workspace_manager_client=workspace_manager_client,
        )
        workspace_manager_client.undeploy_workspace(workspace_id=workspace_id)
        print(f"Workspace {workspace_name} was undeployed successfully")
