from _qwak_proto.qwak.build.v1.build_api_pb2 import (
    GetBuildRequest,
    GetBuildResponse,
    ListBuildsRequest,
    ListBuildsResponse,
    LogPhaseStatusRequest,
    LogPhaseStatusResponse,
    RegisterBuildRequest,
    RegisterBuildResponse,
    RegisterExperimentTrackingRequest,
    RegisterExperimentTrackingResponse,
    RegisterModelSchemaRequest,
    RegisterModelSchemaResponse,
    RegisterTagsRequest,
    RegisterTagsResponse,
    UpdateBuildStatusRequest,
    UpdateBuildStatusResponse,
)
from _qwak_proto.qwak.build.v1.build_api_pb2_grpc import BuildAPIServicer
from _qwak_proto.qwak.build.v1.build_pb2 import Audit, Build, BuildStatus
from google.protobuf.timestamp_pb2 import Timestamp
from qwak_services_mock.mocks.utils.exception_handlers import raise_internal_grpc_error


class BuildOrchestratorBuildApiMock(BuildAPIServicer):
    def __init__(self):
        super(BuildOrchestratorBuildApiMock, self).__init__()
        self._builds: dict[str:Build] = {}

    def given_build(self, build: Build):
        self._builds[build.buildId] = build

    def RegisterBuild(
        self, request: RegisterBuildRequest, context
    ) -> RegisterBuildResponse:
        timestamp = Timestamp()
        timestamp.GetCurrentTime()
        now = timestamp

        audit = Audit(
            created_by="", created_at=now, last_modified_by="", last_modified_at=now
        )
        self._builds[request.buildId] = Build(
            buildId=request.buildId,
            commitId=request.commitId,
            branchId="",
            buildConfig=request.buildConfig,
            build_status=BuildStatus.IN_PROGRESS,
            tags=request.tags,
            steps=request.steps,
            audit=audit,
        )

        return RegisterBuildResponse()

    def UpdateBuildStatus(
        self, request: UpdateBuildStatusRequest, context
    ) -> UpdateBuildStatusResponse:
        self._builds[request.buildId].build_status = request.build_status
        return UpdateBuildStatusResponse()

    def RegisterTags(
        self, request: RegisterTagsRequest, context
    ) -> RegisterTagsResponse:
        self._builds[request.build_id].tags.extend(request.tags)
        return RegisterTagsResponse()

    def RegisterExperimentTracking(
        self, request: RegisterExperimentTrackingRequest, context
    ) -> RegisterExperimentTrackingResponse:
        self._builds[request.build_id].params.update(request.params)
        self._builds[request.build_id].metrics.update(request.metrics)
        return RegisterExperimentTrackingResponse()

    def RegisterModelSchema(
        self, request: RegisterModelSchemaRequest, context
    ) -> RegisterModelSchemaResponse:
        self._builds[request.build_id].model_schema.CopyFrom(request.model_schema)
        return RegisterModelSchemaResponse()

    def GetBuild(self, request: GetBuildRequest, context) -> GetBuildResponse:
        return GetBuildResponse(build=self._builds[request.build_id])

    def ListBuilds(self, request: ListBuildsRequest, context) -> ListBuildsResponse:
        valid_builds = [self._builds[buildId] for buildId in self._builds if self._builds[buildId].model_uuid == request.model_uuid]
        builds_to_return = []

        if len(request.filter.tags) > 0:
            for build in valid_builds:
                if request.filter.require_all_tags:
                    should_be_included = all(
                        tag in build.tags for tag in request.filter.tags
                    )
                else:
                    should_be_included = any(
                        tag in build.tags for tag in request.filter.tags
                    )
                if not request.filter.include_extra_tags:
                    should_be_included = should_be_included and not any(
                        tag not in request.filter.tags for tag in build.tags
                    )
                if should_be_included:
                    builds_to_return.append(build)
        else:
            builds_to_return = valid_builds

        return ListBuildsResponse(build=builds_to_return)

    def LogPhaseStatus(
        self, request: LogPhaseStatusRequest, context
    ) -> LogPhaseStatusResponse:
        try:
            if request.build_id not in self._builds:
                raise Exception(f"Build {request.build_id} doesn't exist")

            return LogPhaseStatusResponse()
        except Exception as e:
            raise_internal_grpc_error(context, e)
