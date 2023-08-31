from typing import List, Optional, Tuple

import grpc
from _qwak_proto.qwak.monitoring.v0.alerting_channel_management_service_pb2 import (
    CreateAlertingChannelRequest,
    DeleteAlertingChannelRequest,
    GetAlertingChannelRequestByName,
    GetAlertingChannelResponseByName,
    ListAlertingChannelRequest,
    ListAlertingChannelResponse,
    UpdateAlertingChannelRequest,
)
from _qwak_proto.qwak.monitoring.v0.alerting_channel_management_service_pb2_grpc import (
    AlertingChannelManagementServiceStub,
)
from _qwak_proto.qwak.monitoring.v0.alerting_channel_pb2 import (
    AlertingChannelCreateOptions,
    AlertingChannelSpec,
    AlertingChannelUpdateOptions,
)

from dependency_injector.wiring import Provide
from google.protobuf.wrappers_pb2 import StringValue

from qwak.clients.alerts_registry.channel import Channel
from qwak.inner.di_configuration import QwakContainer
from qwak.exceptions import QwakException


class AlertingRegistryClient:
    def __init__(self, grpc_channel=Provide[QwakContainer.core_grpc_channel]):
        self.alerting_channel_mgmt_service = AlertingChannelManagementServiceStub(
            grpc_channel
        )

    def create_alerting_channel(self, channel: Channel) -> None:
        try:
            request = CreateAlertingChannelRequest(
                options=AlertingChannelCreateOptions(
                    name=channel.name,
                    spec=AlertingChannelSpec(
                        configuration=channel.channel_conf.to_proto()
                    ),
                )
            )
            self.alerting_channel_mgmt_service.CreateAlertingChannel(request)
        except grpc.RpcError as rpc_error:
            raise QwakException(
                f"Failed to create alerting channel named: {channel.name}. {rpc_error.details()}"
            )
        except Exception as e:
            raise QwakException(
                f"Failed to create alerting channel named: {channel.name}"
            ) from e

    def update_alerting_channel(self, channel_id: str, channel: Channel) -> None:
        try:
            request = UpdateAlertingChannelRequest(
                options=AlertingChannelUpdateOptions(
                    id=channel_id,
                    name=StringValue(value=channel.name),
                    spec=AlertingChannelSpec(
                        configuration=channel.channel_conf.to_proto()
                    ),
                )
            )
            self.alerting_channel_mgmt_service.UpdateAlertingChannel(request)
        except grpc.RpcError as rpc_error:
            raise QwakException(
                f"Failed to update alerting channel named: {channel.name}. \n{rpc_error.details()}"
            )
        except Exception as e:
            raise QwakException(
                f"Failed to update alerting channel named: {channel.name}"
            ) from e

    def get_alerting_channel(self, channel_name: str) -> Optional[Tuple[str, Channel]]:
        try:
            request = GetAlertingChannelRequestByName(name=channel_name)
            response: GetAlertingChannelResponseByName = (
                self.alerting_channel_mgmt_service.GetAlertingChannelByName(request)
            )

            return response.description.metadata.id, Channel.from_proto(
                response.description
            )
        except grpc.RpcError as rpc_error:
            code, name = rpc_error.code().value
            if code == 5:
                return None, None
            else:
                raise QwakException(
                    f"Failed to get alerting channel name: {channel_name}.\n {rpc_error.details()}"
                )
        except Exception as e:
            raise QwakException(
                f"Failed to get alerting channel name: {channel_name}"
            ) from e

    def list_alerting_channel(self) -> List[Tuple[str, Channel]]:
        try:
            response: ListAlertingChannelResponse = (
                self.alerting_channel_mgmt_service.ListAlertingChannel(
                    ListAlertingChannelRequest()
                )
            )
            return [
                (d.metadata.id, Channel.from_proto(d)) for d in response.description
            ]
        except grpc.RpcError as rpc_error:
            raise QwakException(
                f"Failed to list alerting channels\n {rpc_error.details()}"
            )
        except Exception as e:
            raise QwakException("Failed to list alerting channels") from e

    def delete_alerting_channel(self, channel_name: str):
        try:
            self.alerting_channel_mgmt_service.DeleteAlertingChannel(
                DeleteAlertingChannelRequest(name=channel_name)
            )
        except grpc.RpcError as rpc_error:
            raise QwakException(
                f"Failed to delete an alerting channel named: {channel_name}\n {rpc_error.details()}"
            )
        except Exception as e:
            raise QwakException(
                f"Failed to delete an alerting channel named: {channel_name}"
            ) from e
