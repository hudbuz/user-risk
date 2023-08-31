import uuid

from _qwak_proto.qwak.auto_scaling.v1.auto_scaling_service_pb2 import (
    AttachAutoScalingRequest,
    AttachAutoScalingResponse,
)
from _qwak_proto.qwak.auto_scaling.v1.auto_scaling_service_pb2_grpc import (
    AutoScalingServiceServicer,
)
from qwak_services_mock.mocks.utils.exception_handlers import raise_internal_grpc_error


class AutoscalingServiceApiMock(AutoScalingServiceServicer):
    def __init__(self):
        super(AutoscalingServiceApiMock, self).__init__()
        self.autoscaling_policies = dict()

    def AttachAutoScaling(
        self, request: AttachAutoScalingRequest, context
    ) -> AttachAutoScalingResponse:
        try:
            autoscale_id = str(uuid.uuid4())
            self.autoscaling_policies[request.model_id] = (
                autoscale_id,
                request.auto_scaling_config,
            )
            return AttachAutoScalingResponse(auto_scaling_id=autoscale_id)
        except Exception as e:
            raise_internal_grpc_error(context, e)
