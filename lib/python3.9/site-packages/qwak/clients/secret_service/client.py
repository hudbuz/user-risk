from typing import Optional

import grpc
from _qwak_proto.qwak.secret_service.secret_service_pb2 import (
    DeleteSecretRequest,
    GetSecretRequest,
    SetSecretRequest,
)
from _qwak_proto.qwak.secret_service.secret_service_pb2_grpc import SecretServiceStub
from qwak.clients._inner.edge_communications import get_endpoint_url
from qwak.clients.administration.eco_system.client import EcosystemClient
from qwak.exceptions import QwakException
from qwak.inner.tool.grpc.grpc_tools import create_grpc_channel


class SecretServiceClient:
    """
    gRPC client of the secret service
    """

    def __init__(
        self,
        edge_services_url: Optional[str] = None,
        enable_ssl: bool = True,
        grpc_channel: Optional[grpc.Channel] = None,
        environment_id: Optional[str] = None,
    ):
        if not grpc_channel:
            edge_services_url = get_endpoint_url(edge_services_url, environment_id)
            grpc_channel = create_grpc_channel(
                url=edge_services_url,
                enable_ssl=enable_ssl,
                status_for_retry=(
                    grpc.StatusCode.UNAVAILABLE,
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    grpc.StatusCode.INTERNAL,
                ),
                backoff_options={"init_backoff_ms": 250},
            )
        self._secret_service = SecretServiceStub(grpc_channel)

    @staticmethod
    def __get_endpoint_url(endpoint_url=None, environment_id=None):
        if endpoint_url is None:
            user_context = EcosystemClient().get_authenticated_user_context().user
            if environment_id is None:
                environment_id = user_context.account_details.default_environment_id

            if environment_id not in user_context.account_details.environment_by_id:
                raise QwakException(
                    f"Configuration for environment [{environment_id}] was not found"
                )

            endpoint_url = user_context.account_details.environment_by_id[
                environment_id
            ].configuration.edge_services_url

        return endpoint_url

    def get_secret(self, name) -> str:
        try:
            request = GetSecretRequest(name=name)
            return self._secret_service.GetSecret(request).value
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise QwakException(f"Secret [{name}] not found")
            else:
                raise QwakException(
                    f"Failed to get secret, name: [{name}], error code is [{e.code()}], error message is [{e.details()}]"
                )

    def set_secret(self, name, value) -> None:
        try:
            request = SetSecretRequest(name=name, value=value)
            self._secret_service.SetSecret(request)
        except grpc.RpcError as e:
            raise QwakException(
                f"Failed to set secret, name: [{name}], error code is [{e.code()}], error is [{e.details()}]"
            )

    def delete_secret(self, name) -> None:
        try:
            request = DeleteSecretRequest(name=name)
            self._secret_service.DeleteSecret(request)
        except grpc.RpcError as e:
            raise QwakException(
                f"Failed to delete secret, name: [{name}], error is [{e.details()}]"
            )
