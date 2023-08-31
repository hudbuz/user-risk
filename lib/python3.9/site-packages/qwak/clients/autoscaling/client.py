import grpc
from _qwak_proto.qwak.auto_scaling.v1.auto_scaling_pb2 import AutoScalingConfig
from _qwak_proto.qwak.auto_scaling.v1.auto_scaling_service_pb2 import (
    AttachAutoScalingRequest,
    AttachAutoScalingResponse,
)
from _qwak_proto.qwak.auto_scaling.v1.auto_scaling_service_pb2_grpc import (
    AutoScalingServiceStub,
)
from dependency_injector.wiring import Provide
from qwak.exceptions import QwakException
from qwak.inner.di_configuration import QwakContainer


class AutoScalingClient:
    def __init__(self, grpc_channel=Provide[QwakContainer.core_grpc_channel]):
        self._deployment_management = AutoScalingServiceStub(grpc_channel)

    def attach_autoscaling(
        self, model_id: str, variation_name: str, auto_scaling_config: AutoScalingConfig
    ) -> AttachAutoScalingResponse:
        try:
            return self._deployment_management.AttachAutoScaling(
                AttachAutoScalingRequest(
                    model_id=model_id,
                    variation_name=variation_name,
                    auto_scaling_config=auto_scaling_config,
                )
            )
        except grpc.RpcError as e:
            raise QwakException(f"Failed to attach autoscaling, error is {e.details()}")
