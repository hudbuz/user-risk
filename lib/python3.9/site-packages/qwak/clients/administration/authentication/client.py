import grpc
from _qwak_proto.qwak.administration.v0.authentication.authentication_service_pb2 import (
    AuthenticateRequest,
    QwakApiKeyMethod,
)
from _qwak_proto.qwak.administration.v0.authentication.authentication_service_pb2_grpc import (
    AuthenticationServiceStub,
)
from dependency_injector.wiring import Provide
from qwak.exceptions import QwakException
from qwak.inner.di_configuration import QwakContainer


class AuthenticationClient:
    """
    Used for interacting with Qwak's Authentication service
    """

    def __init__(
        self, grpc_channel=Provide[QwakContainer.unauthenticated_core_grpc_channel]
    ):
        self._authentication_service = AuthenticationServiceStub(grpc_channel)

    def authenticate(self, api_key=None):
        request = AuthenticateRequest(
            qwak_api_key_method=QwakApiKeyMethod(qwak_api_key=api_key)
        )
        try:
            return self._authentication_service.Authenticate(request)
        except grpc.RpcError as e:
            raise QwakException(f"Failed to login, error is {e.details()}")
