from _qwak_proto.qwak.kube_deployment_captain.kube_deployment_captain_service_pb2 import (
    DeployFeatureSetResponse,
    DeployStreamingAggregationCompactionResponse,
)
from _qwak_proto.qwak.kube_deployment_captain.kube_deployment_captain_service_pb2_grpc import (
    KubeDeploymentCaptainServicer,
)
from qwak_services_mock.mocks.utils.exception_handlers import raise_internal_grpc_error


class KubeCaptainServiceApiMock(KubeDeploymentCaptainServicer):
    def __init__(self):
        self._last_deploy_streaming_aggregation_compaction_request = None
        super(KubeCaptainServiceApiMock, self).__init__()

    def reset_service(self):
        self._last_deploy_streaming_aggregation_compaction_request = None

    def DeployStreamingAggregationCompaction(self, request, context):
        """deploy streaming aggregation compaction"""
        try:
            self._last_deploy_streaming_aggregation_compaction_request = request
            return DeployStreamingAggregationCompactionResponse()
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def DeployStreamingOfflineFeatureSetV1(self, request, context):
        """deploy streaming offline feature set"""
        try:
            return DeployFeatureSetResponse()
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def DeployBatchFeatureSetV1(self, request, context):
        """deploy batch feature set"""
        try:
            return DeployFeatureSetResponse()
        except Exception as e:
            raise_internal_grpc_error(context, e)
