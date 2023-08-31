import uuid

from _qwak_proto.qwak.monitoring.v0.alerting_channel_management_service_pb2 import CreateAlertingChannelRequest, \
    CreateAlertingChannelResponse, GetAlertingChannelRequestByName, GetAlertingChannelResponseByName, \
    DeleteAlertingChannelRequest, ListAlertingChannelRequest, ListAlertingChannelResponse, DeleteAlertingChannelResponse
from _qwak_proto.qwak.monitoring.v0.alerting_channel_management_service_pb2_grpc import \
    AlertingChannelManagementServiceServicer
from _qwak_proto.qwak.monitoring.v0.alerting_channel_pb2 import AlertingChannelDescription, AlertingChannelMetadata
from qwak_services_mock.mocks.utils.exception_handlers import raise_internal_grpc_error


class AlertsRegistryServiceApiMock(AlertingChannelManagementServiceServicer):
    def __init__(self):
        super(AlertsRegistryServiceApiMock, self).__init__()
        self.alerts_channels = dict()

    def CreateAlertingChannel(self, request: CreateAlertingChannelRequest, context):
        metadata = AlertingChannelMetadata(id=str(uuid.uuid4()),name=request.options.name)
        self.alerts_channels[request.options.name] = AlertingChannelDescription(spec=request.options.spec,
                                                                                metadata=metadata)
        return CreateAlertingChannelResponse()

    def GetAlertingChannelByName(self, request: GetAlertingChannelRequestByName, context):
        try:
            return GetAlertingChannelResponseByName(
                description=self.alerts_channels[request.name])
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def ListAlertingChannel(self, request: ListAlertingChannelRequest, context):
        try:
            return ListAlertingChannelResponse(
                audience_entries=[
                    desc
                    for name, desc in self.audiences.items()
                ]
            )
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def DeleteAlertingChannel(self, request: DeleteAlertingChannelRequest, context):
        try:
            del self.alerts_channels[request.name]
            return DeleteAlertingChannelResponse()
        except Exception as e:
            raise_internal_grpc_error(context, e)
