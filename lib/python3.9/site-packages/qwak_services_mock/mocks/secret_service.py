import grpc
from _qwak_proto.qwak.secret_service.secret_service_pb2 import (
    DeleteSecretRequest,
    DeleteSecretResponse,
    GetSecretRequest,
    GetSecretResponse,
    SetSecretRequest,
    SetSecretResponse,
)
from _qwak_proto.qwak.secret_service.secret_service_pb2_grpc import (
    SecretServiceServicer,
)


class SecretServiceMock(SecretServiceServicer):
    secrets = {}

    def SetSecret(self, request: SetSecretRequest, context) -> SetSecretResponse:
        """Set a value for a secret"""
        self.secrets[request.name] = request.value
        return SetSecretResponse()

    def GetSecret(self, request: GetSecretRequest, context) -> GetSecretResponse:
        """Get a secret value"""
        if request.name in self.secrets:
            return GetSecretResponse(
                name=request.name, value=self.secrets[request.name]
            )
        context.set_code(grpc.StatusCode.NOT_FOUND)
        context.set_details("Secret not found")
        return GetSecretResponse()

    def DeleteSecret(
        self, request: DeleteSecretRequest, context
    ) -> DeleteSecretResponse:
        """Remove a secret"""
        if request.name in self.secrets:
            del self.secrets[request.name]
            return DeleteSecretResponse()
        context.set_code(grpc.StatusCode.NOT_FOUND)
        context.set_details("Secret not found")
        return DeleteSecretResponse()
