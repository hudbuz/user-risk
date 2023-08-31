import grpc
from _qwak_proto.qwak.builds.builds_pb2 import (
    BuildStatus,
    ExperimentTrackingValues,
    GetBuildRequest,
    GetBuildResponse,
    ListBuildsRequest,
    ListBuildsResponse,
    Metric,
    Param,
    RegisterBuildRequest,
    RegisterExperimentTrackingValuesRequest,
    RegisterModelSchemaRequest,
    UpdateBuildStatusRequest,
)
from _qwak_proto.qwak.builds.builds_pb2_grpc import BuildsManagementServiceStub
from dependency_injector.wiring import Provide
from qwak.exceptions import QwakException
from qwak.inner.di_configuration import QwakContainer


class BuildsManagementClient:
    """
    Used for interacting with Feature Registry endpoints
    """

    def __init__(self, grpc_channel=Provide[QwakContainer.core_grpc_channel]):
        self._builds_management_service = BuildsManagementServiceStub(grpc_channel)

    def register_build(self, build_spec, model_schema=None, qwak_calling_user_id=""):
        try:
            self._builds_management_service.RegisterBuild(
                RegisterBuildRequest(
                    build_spec=build_spec,
                    model_schema=model_schema,
                    qwak_calling_user_id=qwak_calling_user_id,
                )
            )

        except grpc.RpcError as e:
            raise QwakException(f"Failed to register build, error is {e.details()}")

    def update_build_status(self, build_id, build_status, qwak_user_id=""):
        try:
            self._builds_management_service.UpdateBuildStatus(
                UpdateBuildStatusRequest(
                    build_id=build_id,
                    build_status=BuildStatus.Value(build_status),
                    qwak_calling_user_id=qwak_user_id,
                )
            )

        except grpc.RpcError as e:
            raise QwakException(
                f"Failed to update build status, error is {e.details()}"
            )

    def get_build(self, build_id) -> GetBuildResponse:
        try:
            return self._builds_management_service.GetBuild(
                GetBuildRequest(build_id=build_id)
            )
        except grpc.RpcError as e:
            raise QwakException(f"Failed to get build, error is {e.details()}")

    def is_build_exists(self, build_id: str) -> bool:
        try:
            self._builds_management_service.GetBuild(GetBuildRequest(build_id=build_id))
            return True
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                return False
            raise QwakException(
                f"Failed to check if build {build_id} is exists, error is {e.details()}"
            )

    def register_schema(self, build_id, model_schema):
        try:
            return self._builds_management_service.RegisterModelSchema(
                RegisterModelSchemaRequest(build_id=build_id, model_schema=model_schema)
            )

        except grpc.RpcError as e:
            raise QwakException(f"Failed to register schema, error is {e.details()}")

    def register_experiment_tracking_values(
        self, build_id, qwak_params=None, cli_params=None, qwak_metrics=None
    ):
        qwak_params = qwak_params if qwak_params else {}
        cli_params = cli_params if cli_params else {}
        qwak_metrics = qwak_metrics if qwak_metrics else {}

        try:
            params = cli_params.copy()
            params.update(qwak_params)

            experiment_tracking_values = ExperimentTrackingValues(
                metrics=[
                    Metric(key=key, value=value) for key, value in qwak_metrics.items()
                ],
                params=[
                    Param(key=key, value=str(value)) for key, value in params.items()
                ],
            )

            return self._builds_management_service.RegisterExperimentTrackingValues(
                RegisterExperimentTrackingValuesRequest(
                    build_id=build_id,
                    experiment_tracking_values=experiment_tracking_values,
                )
            )

        except grpc.RpcError as e:
            raise QwakException(
                f"Failed to register experiment tracking values, error is {e.details()}"
            )

    def list_builds(self, model_uuid="", **kwargs) -> ListBuildsResponse:
        try:
            _model_uuid = model_uuid if model_uuid else kwargs.get("branch_id")
            if not _model_uuid:
                raise QwakException("missing argument model uuid or branch id.")
            return self._builds_management_service.ListBuilds(
                ListBuildsRequest(model_uuid=_model_uuid)
            )
        except grpc.RpcError as e:
            raise QwakException(f"Failed to list builds, error is {e.details()}")
