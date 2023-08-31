import grpc
from _qwak_proto.qwak.user_application.common.v0.resources_pb2 import (
    ClientPodComputeResources,
    PodComputeResourceTemplateSpec,
)
from _qwak_proto.qwak.workspace.workspace_pb2 import WorkspaceSpec
from _qwak_proto.qwak.workspace.workspace_service_pb2 import (
    CreateWorkspaceRequest,
    CreateWorkspaceResponse,
    DeleteWorkspaceRequest,
    DeployWorkspaceRequest,
    DeployWorkspaceResponse,
    GetWorkspaceByIdRequest,
    GetWorkspaceByIdResponse,
    ListWorkspaceImagesRequest,
    ListWorkspaceImagesResponse,
    ListWorkspacesRequest,
    ListWorkspacesResponse,
    UndeployWorkspaceRequest,
    UpdateWorkspaceRequest,
)
from _qwak_proto.qwak.workspace.workspace_service_pb2_grpc import (
    WorkspaceManagementServiceStub,
)
from dependency_injector.wiring import Provide
from qwak.exceptions import QwakException
from qwak.inner.di_configuration import QwakContainer


class WorkspaceManagerClient:
    """
    Used for interacting with Workspace Manager endpoints
    """

    def __init__(self, grpc_channel=Provide[QwakContainer.core_grpc_channel]):
        self._workspace_manager = WorkspaceManagementServiceStub(grpc_channel)

    def create_workspace(
        self, workspace_name: str, image_id: str, template_id: str
    ) -> CreateWorkspaceResponse:
        """
        Args:
            workspace_name: The name of the workspace
            image_id: The image id of the workspace deployment
            template_id: The id of the template to use for the workspace deployment

        Returns:
            The response of the created workspace
        """
        try:
            create_workspace_request = CreateWorkspaceRequest(
                workspace_spec=WorkspaceSpec(
                    workspace_name=workspace_name,
                    image_id=image_id,
                    client_pod_compute_resources=ClientPodComputeResources(
                        template_spec=PodComputeResourceTemplateSpec(
                            template_id=template_id
                        )
                    ),
                )
            )
            return self._workspace_manager.CreateWorkspace(create_workspace_request)
        except grpc.RpcError as e:
            raise QwakException(
                f"Failed to create workspace,  code: {e.code().name} error is {e.details()}"
            )

    def update_workspace(
        self,
        workspace_id: str,
        workspace_name: str = "",
        image_id: str = "",
        template_id: str = "",
    ) -> CreateWorkspaceResponse:
        """
        Args:
            workspace_id: The id of the workspace
            workspace_name: The name of the workspace
            image_id: The image id of the workspace deployment
            template_id: The id of the template to use for the workspace deployment
        Returns:
            The response of the updated workspace
        """
        try:
            update_workspace_request = UpdateWorkspaceRequest(
                workspace_id=workspace_id,
                workspace_spec=WorkspaceSpec(
                    workspace_name=workspace_name,
                    image_id=image_id,
                    client_pod_compute_resources=ClientPodComputeResources(
                        template_spec=PodComputeResourceTemplateSpec(
                            template_id=template_id
                        )
                    ),
                ),
            )
            return self._workspace_manager.UpdateWorkspace(update_workspace_request)
        except grpc.RpcError as e:
            raise QwakException(
                f"Failed to update workspace,  code: {e.code().name} error is {e.details()}"
            )

    def deploy_workspace(self, workspace_id: str) -> DeployWorkspaceResponse:
        """
        Args:
            workspace_id: The id of the workspace
        Returns:
            The response of the deployed workspace
        """
        try:
            deploy_workspace_request = DeployWorkspaceRequest(workspace_id=workspace_id)
            return self._workspace_manager.DeployWorkspace(deploy_workspace_request)
        except grpc.RpcError as e:
            raise QwakException(
                f"Failed to deploy workspace,  code: {e.code().name} error is {e.details()}"
            )

    def get_workspaces(self) -> ListWorkspacesResponse:
        """
        Returns:
            The response of the list workspaces
        """
        try:
            return self._workspace_manager.ListWorkspaces(ListWorkspacesRequest())
        except grpc.RpcError as e:
            raise QwakException(
                f"Failed to list workspaces,  code: {e.code().name} error is {e.details()}"
            )

    def get_workspace_by_id(self, workspace_id: str) -> GetWorkspaceByIdResponse:
        """
        Args:
            workspace_id: The id of the workspace
        Returns:
            The response of the workspace
        """
        try:
            return self._workspace_manager.GetWorkspaceById(
                GetWorkspaceByIdRequest(workspace_id=workspace_id)
            )
        except grpc.RpcError as e:
            raise QwakException(
                f"Failed to get workspace by id,  code: {e.code().name} error is {e.details()}"
            )

    def delete_workspace(self, workspace_id: str) -> DeployWorkspaceResponse:
        """
        Args:
            workspace_id: The id of the workspace
        Returns:
            The response of the deleted workspace
        """
        try:
            delete_workspace_request = DeleteWorkspaceRequest(workspace_id=workspace_id)
            return self._workspace_manager.DeleteWorkspace(delete_workspace_request)
        except grpc.RpcError as e:
            raise QwakException(
                f"Failed to delete workspace, code: {e.code().name} error is {e.details()}"
            )

    def undeploy_workspace(self, workspace_id: str) -> DeployWorkspaceResponse:
        """
        Args:
            workspace_id: The id of the workspace
        Returns:
            The response of the undeployed workspace
        """
        try:
            undeploy_workspace_request = UndeployWorkspaceRequest(
                workspace_id=workspace_id
            )
            return self._workspace_manager.UndeployWorkspace(undeploy_workspace_request)
        except grpc.RpcError as e:
            raise QwakException(
                f"Failed to undeploy workspace,  code: {e.code().name} error is {e.details()}"
            )

    def get_workspace_images(self) -> ListWorkspaceImagesResponse:
        """
        Returns:
            The response of the list workspace image
        """
        try:
            return self._workspace_manager.ListWorkspaceImages(
                ListWorkspaceImagesRequest()
            )
        except grpc.RpcError as e:
            raise QwakException(
                f"Failed to list workspace image types, code: {e.code().name} error is {e.details()}"
            )
