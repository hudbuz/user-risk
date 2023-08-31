import grpc
from _qwak_proto.qwak.deployment.alert_pb2 import (
    NotificationChannel,
    NotificationChannelSettings,
)
from _qwak_proto.qwak.deployment.alert_service_pb2 import (
    ApplyNotificationChannelRequest,
    DeleteNotificationChannelRequest,
    GetNotificationChannelDetailsListRequest,
)
from _qwak_proto.qwak.deployment.alert_service_pb2_grpc import (
    AlertManagementServiceStub,
)
from dependency_injector.wiring import Provide
from qwak.exceptions import QwakException
from qwak.inner.di_configuration import QwakContainer


class AlertsManagementClient:
    def __init__(self, grpc_channel=Provide[QwakContainer.core_grpc_channel]):
        self._alerts_management_service = AlertManagementServiceStub(grpc_channel)

    def configure_channel(
        self, channel_name: str, notification_settings: NotificationChannelSettings
    ):
        try:
            self._alerts_management_service.ApplyNotificationChannel(
                ApplyNotificationChannelRequest(
                    notification_channel=NotificationChannel(
                        notification_channel_name=channel_name,
                        notification_settings=notification_settings,
                    ),
                )
            )

        except grpc.RpcError as e:
            raise QwakException(
                f"Failed to save notification channel, error is {e.details()}"
            )

    def list_channels(self):
        try:
            return self._alerts_management_service.GetNotificationChannelDetailsList(
                GetNotificationChannelDetailsListRequest()
            )

        except grpc.RpcError as e:
            raise QwakException(
                f"Failed to fetch notification channels, error is {e.details()}"
            )

    def delete_channel(self, notification_channel_id):
        try:
            self._alerts_management_service.DeleteNotificationChannel(
                DeleteNotificationChannelRequest(
                    notification_channel_id=notification_channel_id
                )
            )

        except grpc.RpcError as e:
            raise QwakException(
                f"Failed to delete notification channel, error is {e.details()}"
            )
