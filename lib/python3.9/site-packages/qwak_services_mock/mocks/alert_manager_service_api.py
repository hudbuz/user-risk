import uuid

from _qwak_proto.qwak.deployment.alert_service_pb2 import (
    ApplyNotificationChannelRequest,
    ApplyNotificationChannelResponse,
    DeleteNotificationChannelRequest,
    DeleteNotificationChannelResponse,
    GetNotificationChannelDetailsListRequest,
    GetNotificationChannelDetailsListResponse,
)
from _qwak_proto.qwak.deployment.alert_service_pb2_grpc import (
    AlertManagementServiceServicer,
)
from qwak_services_mock.mocks.utils.exception_handlers import raise_internal_grpc_error


class AlertManagerServiceApiMock(AlertManagementServiceServicer):
    def __init__(self):
        super(AlertManagerServiceApiMock, self).__init__()
        self.notification_channels = dict()

    def ApplyNotificationChannel(
        self, request: ApplyNotificationChannelRequest, context
    ) -> ApplyNotificationChannelResponse:
        """Channels to add or update"""
        try:
            request.notification_channel.notification_channel_id = str(uuid.uuid4())
            self.notification_channels[
                request.notification_channel.notification_channel_id
            ] = request.notification_channel
            return ApplyNotificationChannelResponse()
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def DeleteNotificationChannel(
        self, request: DeleteNotificationChannelRequest, context
    ) -> DeleteNotificationChannelResponse:
        """Delete notification channel"""
        try:
            del self.notification_channels[request.notification_channel_id]
            return DeleteNotificationChannelResponse()
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def GetNotificationChannelDetailsList(
        self, request: GetNotificationChannelDetailsListRequest, context
    ) -> GetNotificationChannelDetailsListResponse:
        """Get Full Details for channels"""
        try:
            return GetNotificationChannelDetailsListResponse(
                notification_channels=list(self.notification_channels.values())
            )
        except Exception as e:
            raise_internal_grpc_error(context, e)
