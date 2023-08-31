import grpc
from _qwak_proto.qwak.administration.v0.environments.environment_pb2 import (
    QwakEnvironmentStatus,
)
from _qwak_proto.qwak.administration.v0.environments.environment_service_pb2 import (
    GetEnvironmentApplicationUserCredentialsRequest,
    GetEnvironmentApplicationUserCredentialsResponse,
    ListEnvironmentsRequest,
    ListEnvironmentsResponse,
)
from _qwak_proto.qwak.administration.v0.environments.environment_service_pb2_grpc import (
    EnvironmentServiceStub,
)
from dependency_injector.wiring import Provide
from qwak.exceptions import QwakException
from qwak.inner.di_configuration import QwakContainer


class EnvironmentClient:
    """
    Used for interacting with Qwak's Environemt
    """

    def __init__(self, grpc_channel=Provide[QwakContainer.core_grpc_channel]):
        self._environment_service = EnvironmentServiceStub(grpc_channel)

    def list_environments(self) -> ListEnvironmentsResponse:
        """
        List of all environment without filter
        Return ListEnvironmentsResponse
        """
        request = ListEnvironmentsRequest()
        try:
            return self._environment_service.ListEnvironments(request)
        except grpc.RpcError as e:
            raise QwakException(
                f"Failed to get List of environments, error is {e.details()}"
            )

    def list_environments_by_status(
        self, status: QwakEnvironmentStatus
    ) -> ListEnvironmentsResponse:
        """
        List of all environment without filter
        Args: status which filter environment by
        Return: ListEnvironmentsResponse

        """
        try:
            request = ListEnvironmentsRequest(
                filter=ListEnvironmentsRequest.Filter(environment_status=status)
            )
            return self._environment_service.ListEnvironments(request)
        except grpc.RpcError as e:
            raise QwakException(
                f"Failed to get List of environments with status, error is {e.details()}"
            )

    def get_environment_application_user(
        self, environment_id: str
    ) -> GetEnvironmentApplicationUserCredentialsResponse:
        """
        Get application user by environment id
        Return: GetEnvironmentApplicationUserCredentialsResponse
        """
        request = GetEnvironmentApplicationUserCredentialsRequest(
            environment_id=environment_id
        )
        try:
            return self._environment_service.GetEnvironmentApplicationUserCredentials(
                request
            )
        except grpc.RpcError as e:
            raise QwakException(
                f"Failed to get application user, error is {e.details()}"
            )
