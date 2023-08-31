from collections import defaultdict

import grpc
from _qwak_proto.qwak.builds.builds_pb2 import (
    SUCCESSFUL,
    Build,
    BuildSpec,
    GetBuildResponse,
    ListBuildsResponse,
    RegisterBuildResponse,
    RegisterExperimentTrackingValuesResponse,
    RegisterModelSchemaResponse,
    UpdateBuildStatusResponse,
)
from _qwak_proto.qwak.builds.builds_pb2_grpc import BuildsManagementServiceServicer


class BuildsManagementServiceMock(BuildsManagementServiceServicer):
    def __init__(self):
        self.builds = defaultdict(lambda: {})
        self.builds_to_models = defaultdict()
        self.init_status = SUCCESSFUL

    def RegisterBuild(self, request, context):
        spec = request.build_spec
        self.builds[spec.model_uuid] = {
            spec.build_id: {"spec": spec, "status": self.init_status}
        }
        self.builds_to_models[spec.build_id] = spec.model_uuid

        return RegisterBuildResponse(
            build_id=spec.build_id, build_status=self.init_status
        )

    def UpdateBuildStatus(self, request, context):
        model_uuid = self.builds_to_models[request.build_id]
        if model_uuid in self.builds:
            self.builds[model_uuid][request.build_id]["status"] = str(
                request.build_status
            )
        else:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Received a non existing build id")

        return UpdateBuildStatusResponse()

    def RegisterModelSchema(self, request, context):
        model_uuid = self.builds_to_models[request.build_id]
        if model_uuid in self.builds:
            self.builds[model_uuid][request.build_id]["schema"] = request.model_schema
        return RegisterModelSchemaResponse()

    def RegisterExperimentTrackingValues(self, request, context):
        model_uuid = self.builds_to_models[request.build_id]
        if model_uuid in self.builds:
            spec = BuildSpec(
                build_id=request.build_id,
                experiment_tracking_values=request.experiment_tracking_values,
            )
            self.builds[model_uuid][request.build_id] = {
                "spec": spec,
                "status": 1,
            }
        else:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Received a non existing build id")
        return RegisterExperimentTrackingValuesResponse()

    def GetBuild(self, request, context):
        model_uuid = self.builds_to_models[request.build_id]
        if model_uuid in self.builds:
            persistent_build = self.builds[model_uuid][request.build_id]

            build = self._persistent_build_to_build_proto(persistent_build)

            return GetBuildResponse(build=build)
        else:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Received a non existing build id")
            return GetBuildResponse()

    def ListBuilds(self, request, context):
        return ListBuildsResponse(
            builds=[
                self._persistent_build_to_build_proto(persistent_build)
                for persistent_build in self.builds[request.model_uuid].values()
            ]
        )

    @staticmethod
    def _persistent_build_to_build_proto(persistent_build):
        spec = persistent_build["spec"]
        return Build(
            build_spec=BuildSpec(
                build_id=spec.build_id,
                commit_id=spec.commit_id,
                model_id=spec.model_id,
                branch_name=spec.branch_name,
                tags=spec.tags,
                experiment_tracking_values=spec.experiment_tracking_values,
            ),
            build_status=SUCCESSFUL,
            model_schema=persistent_build["schema"]
            if "schema" in persistent_build
            else None,
        )
