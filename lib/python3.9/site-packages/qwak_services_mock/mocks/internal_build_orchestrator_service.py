from _qwak_proto.qwak.builds.internal_builds_orchestrator_service_pb2 import (
    BuildInitDetails,
    InternalBuildModelRequest,
    InternalBuildModelResponse,
)
from _qwak_proto.qwak.builds.internal_builds_orchestrator_service_pb2_grpc import (
    InternalBuildsOrchestratorServiceServicer,
)
from qwak_services_mock.mocks.utils.exception_handlers import raise_internal_grpc_error


class InternalBuildOrchestratorServiceMock(InternalBuildsOrchestratorServiceServicer):
    def __init__(self):
        super(InternalBuildOrchestratorServiceMock, self).__init__()
        self._builds_requests: dict[str:BuildInitDetails] = {}

    def BuildModel(self, request: InternalBuildModelRequest, context):
        """Build a serving model image"""
        try:
            self._builds_requests[
                request.build_init_details.build_spec.build_properties.build_id
            ] = request.build_init_details
            return InternalBuildModelResponse()
        except Exception as e:
            raise_internal_grpc_error(context, e)
