from _qwak_proto.qwak.self_service.user.v1.user_service_pb2 import (
    GenerateApiKeyRequest,
    GenerateApiKeyResponse,
    RevokeApiKeyRequest,
    RevokeApiKeyResponse,
)
from _qwak_proto.qwak.self_service.user.v1.user_service_pb2_grpc import UserServiceStub
from dependency_injector.wiring import Provide
from grpc import RpcError, StatusCode
from qwak.exceptions import QwakException
from qwak.inner.di_configuration import QwakContainer

APIKEY_ALREADY_EXISTS = "Api-key already exists"


class SelfServiceUserClient:
    """
    Used for interacting with Qwak's Self service -  user service
    """

    def __init__(self, grpc_channel=Provide[QwakContainer.core_grpc_channel]):
        self._user_service = UserServiceStub(grpc_channel)

    def generate_apikey(
        self,
        user_id: str,
        environment_id: str,
        force: bool = False,
    ) -> GenerateApiKeyResponse:
        """
        Generate api key
        user_id: the wanted user id.
        environment_id: the wanted environment id
        force: if override existing one.
        Return Api Key for the wanted user and environment.
        """
        request = GenerateApiKeyRequest(
            user_id=user_id, environment_id=environment_id, force=force
        )
        try:
            return self._user_service.GenerateApiKey(request)
        except RpcError as e:
            if e.code() == StatusCode.PERMISSION_DENIED:
                raise QwakException(
                    f"You are not authorized to perform administration operations. error is  {e.details()}"
                )
            elif e.code() == StatusCode.ALREADY_EXISTS:
                raise QwakException(APIKEY_ALREADY_EXISTS)
            raise QwakException(f"Failed to generate apikey, error is {e.details()}")

    def revoke_apikey(self, user_id: str, environment_id: str) -> RevokeApiKeyResponse:
        """
        Revoke api eky
        user_id: the wanted user id to revoke from
        environment_id: the wanted environment id
        Return if the api key has been revoked

        """
        try:
            request = RevokeApiKeyRequest(
                user_id=user_id, environment_id=environment_id
            )
            return self._user_service.RevokeApiKey(request)

        except RpcError as e:
            if e.code() == StatusCode.PERMISSION_DENIED:
                raise QwakException(
                    f"You are not authorized to perform administration operations. error is  {e.details()}"
                )
            raise QwakException(f"Failed to generate apikey, error is {e.details()}")
