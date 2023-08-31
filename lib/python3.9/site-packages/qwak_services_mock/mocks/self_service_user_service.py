from typing import Dict

from _qwak_proto.qwak.self_service.user.v1.user_service_pb2 import (
    GenerateApiKeyResponse,
    RevokeApiKeyResponse,
)
from _qwak_proto.qwak.self_service.user.v1.user_service_pb2_grpc import (
    UserServiceServicer,
)

API_KEY_FORMAT = "apikey-for-{}-{}"


class SelfServiceUserServiceMock(UserServiceServicer):
    def __init__(self):
        super(SelfServiceUserServiceMock, self).__init__()
        self.apikeys: Dict[(str, str), str] = dict()

    def RevokeApiKey(self, request, context) -> RevokeApiKeyResponse:
        if self.apikeys.get((request.user_id, request.environment_id)):
            self.apikeys.pop((request.user_id, request.environment_id))
            return RevokeApiKeyResponse(status=RevokeApiKeyResponse.Status.REVOKED)
        else:
            return RevokeApiKeyResponse(
                status=RevokeApiKeyResponse.Status.INVALID_USER_STATUS
            )

    def GenerateApiKey(self, request, context) -> GenerateApiKeyResponse:
        api_key = API_KEY_FORMAT.format(request.user_id, request.environment_id)
        self.apikeys[(request.user_id, request.environment_id)] = api_key
        return GenerateApiKeyResponse(api_key=api_key)
