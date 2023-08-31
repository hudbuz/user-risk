import grpc
from _qwak_proto.qwak.administration.authenticated_user.v1.authenticated_user_service_pb2 import (
    GetCloudCredentialsRequest,
    GetCloudCredentialsResponse,
    GetDetailsRequest,
    GetDetailsResponse,
)
from _qwak_proto.qwak.administration.authenticated_user.v1.authenticated_user_service_pb2_grpc import (
    AuthenticatedUserStub,
)
from dependency_injector.wiring import Provide
from qwak.exceptions import QwakException
from qwak.inner.di_configuration import QwakContainer


class AuthenticatedUserClient:
    """
    Used for interacting with Qwaks's Authenticated User service
    """

    def __init__(self, grpc_channel=Provide[QwakContainer.core_grpc_channel]):
        self._authenticated_user = AuthenticatedUserStub(grpc_channel)

    def get_details(self) -> GetDetailsResponse:
        request = GetDetailsRequest()
        try:
            return self._authenticated_user.GetDetails(request)
        except grpc.RpcError as e:
            raise QwakException(
                f"Failed to get authenticated user details, error is {e.details()}"
            )

    def get_cloud_credentials(self) -> GetCloudCredentialsResponse:
        request = GetCloudCredentialsRequest()
        try:
            return self._authenticated_user.GetCloudCredentials(request)
        except grpc.RpcError as e:
            raise QwakException(
                f"Failed to get cloud credentials, error is {e.details()}"
            )
