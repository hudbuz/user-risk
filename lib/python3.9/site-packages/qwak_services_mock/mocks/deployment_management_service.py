import uuid
from collections import defaultdict
from typing import Dict, Optional

from _qwak_proto.qwak.deployment.deployment_pb2 import (
    DeploymentDetails,
    EnvironmentApplyModelTrafficResponse,
    EnvironmentDeploymentDetailsMessage,
    EnvironmentDeploymentMessage,
    EnvironmentDeploymentResultMessage,
    EnvironmentRuntimeDeploymentSettingsResultMessage,
    EnvironmentTrafficMessage,
    EnvironmentUndeploymentMessage,
    EnvironmentUndeploymentResultMessage,
    KubeDeploymentType,
    ModelDeploymentStatus,
    Variation,
)
from _qwak_proto.qwak.deployment.deployment_service_pb2 import (
    ApplyModelTrafficRequest,
    ApplyModelTrafficResponse,
    DeployModelRequest,
    DeployModelResponse,
    GetDeploymentDetailsRequest,
    GetDeploymentDetailsResponse,
    GetDeploymentStatusRequest,
    GetDeploymentStatusResponse,
    GetModelTrafficRequest,
    GetModelTrafficResponse,
    UndeployModelRequest,
    UndeployModelResponse,
    UpdateDeploymentRuntimeSettingsRequest,
    UpdateDeploymentRuntimeSettingsResponse,
)
from _qwak_proto.qwak.deployment.deployment_service_pb2_grpc import (
    DeploymentManagementServiceServicer,
)
from google.protobuf.timestamp_pb2 import Timestamp
from qwak_services_mock.mocks.utils.exception_handlers import raise_internal_grpc_error

FAILED_INITIATING_DEPLOYMENT_MESSAGE = "Failed initiating deployment"
DEPLOYMENT_ALREADY_RUNNING_MESSAGE = "There is already an active deployment"
FAILED_TO_UNDEPLOY_NO_ACTIVE_DEPLOYMENT = "No Active Deployment Found"
FAILED_TO_UNDEPLOY_DEPLOYMENT_IN_PROGRESS = "Deployment In Progress"
NO_ACTIVE_TRAFFIC = "No active traffic found to update"
DEFAULT_VARIATION_NAME = "default"


class DeploymentManagementServiceMock(DeploymentManagementServiceServicer):
    def __init__(self):
        self.active_deployments = defaultdict(dict)
        self.deployments_history = defaultdict(list)
        self.model_traffic = dict()
        self.model_uuid_to_active_deployment_ids = defaultdict(list)
        self.models_in_progress = set()
        self.__deployment_to_fail_on_init = set()

    def DeployModel(self, request: DeployModelRequest, context) -> DeployModelResponse:
        try:
            environments = request.environment_to_deployment.keys()
            results = dict()
            for env in environments:
                env_request_details: EnvironmentDeploymentMessage = (
                    request.environment_to_deployment[env]
                )
                if self.__should_fail_deployment(
                    env_request_details.model_id, env_request_details.build_id
                ):
                    results[env] = EnvironmentDeploymentResultMessage(
                        deployment_named_id="",
                        status=ModelDeploymentStatus.FAILED_INITIATING_DEPLOYMENT,
                        info=FAILED_INITIATING_DEPLOYMENT_MESSAGE,
                    )
                elif self.__deployment_in_progress(env_request_details.model_id, env):
                    results[env] = EnvironmentDeploymentResultMessage(
                        deployment_named_id="",
                        status=ModelDeploymentStatus.FAILED_INITIATING_DEPLOYMENT,
                        info=DEPLOYMENT_ALREADY_RUNNING_MESSAGE,
                    )
                else:
                    deployment_id = str(uuid.uuid4())
                    variation_name = self.__get_variation_name(
                        env_request_details.hosting_service
                    )
                    results[env] = EnvironmentDeploymentResultMessage(
                        deployment_named_id=deployment_id,
                        status=ModelDeploymentStatus.INITIATING_DEPLOYMENT,
                        info="",
                    )
                    self.__mark_deployment_in_progress(
                        env_request_details.model_id, env
                    )
                    deployment_key = self.get_deployment_key(
                        env_request_details.model_id, variation_name, env
                    )
                    if deployment_key in self.active_deployments:
                        prev_deployment = self.active_deployments[deployment_key]
                        prev_deployment[
                            "status"
                        ] = ModelDeploymentStatus.SUCCESSFUL_UNDEPLOYMENT
                        self.deployments_history[env_request_details.model_uuid].insert(
                            0, prev_deployment
                        )
                        self.model_uuid_to_active_deployment_ids[
                            env_request_details.model_uuid
                        ].remove(prev_deployment["deployment_id"])
                    self.active_deployments[deployment_key] = {
                        "env": env,
                        "model_id": env_request_details.model_id,
                        "build_id": env_request_details.build_id,
                        "model_uuid": env_request_details.model_uuid,
                        "deployment_id": deployment_id,
                        "hosting_service": env_request_details.hosting_service,
                        "timestamp": Timestamp(),
                        "status": ModelDeploymentStatus.INITIATING_DEPLOYMENT,
                    }
                    self.model_uuid_to_active_deployment_ids[
                        env_request_details.model_uuid
                    ].append(deployment_id)
                    if (
                        env_request_details.hosting_service.kube_deployment.kube_deployment_type
                        == KubeDeploymentType.ONLINE
                    ):
                        self.model_traffic[
                            (env_request_details.model_id, env)
                        ] = (
                            env_request_details.hosting_service.kube_deployment.serving_strategy.realtime_config.traffic_config
                        )
                    else:
                        try:
                            del self.model_traffic[(env_request_details.model_id, env)]
                        except Exception:
                            pass

            return DeployModelResponse(environment_to_deployment_result=results)
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def UndeployModel(
        self, request: UndeployModelRequest, context
    ) -> UndeployModelResponse:
        try:
            environments = request.environment_to_undeployment.keys()
            results = dict()
            for env in environments:
                details: EnvironmentUndeploymentMessage = (
                    request.environment_to_undeployment[env]
                )
                deployment_key = self.get_deployment_key(
                    details.model_id, details.variation_name, env
                )
                if deployment_key not in self.active_deployments:
                    results[env] = EnvironmentUndeploymentResultMessage(
                        status=ModelDeploymentStatus.FAILED_UNDEPLOYMENT,
                        info=FAILED_TO_UNDEPLOY_NO_ACTIVE_DEPLOYMENT,
                    )
                elif self.__deployment_in_progress(details.model_id, env):
                    results[env] = EnvironmentUndeploymentResultMessage(
                        status=ModelDeploymentStatus.FAILED_UNDEPLOYMENT,
                        info=FAILED_TO_UNDEPLOY_DEPLOYMENT_IN_PROGRESS,
                    )
                else:
                    deployment_details = self.active_deployments[deployment_key]
                    deployment_details[
                        "status"
                    ] = ModelDeploymentStatus.SUCCESSFUL_UNDEPLOYMENT
                    self.deployments_history[details.model_uuid].insert(
                        0, deployment_details
                    )
                    self.model_uuid_to_active_deployment_ids[details.model_uuid].remove(
                        deployment_details["deployment_id"]
                    )
                    del self.active_deployments[deployment_key]
                    if details.traffic_config:
                        self.model_traffic[
                            (details.model_id, env)
                        ] = details.traffic_config
                    else:
                        try:
                            del self.model_traffic[(details.model_id, env)]
                        except Exception:
                            pass
                    results[env] = EnvironmentUndeploymentResultMessage(
                        status=ModelDeploymentStatus.SUCCESSFUL_UNDEPLOYMENT, info=""
                    )

            return UndeployModelResponse(environment_to_undeployment_result=results)
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def GetDeploymentStatus(
        self, request: GetDeploymentStatusRequest, context
    ) -> GetDeploymentStatusResponse:
        try:
            deployment = self.get_deployment_by_id(request.deployment_named_id)
            status = (
                deployment["status"]
                if deployment
                else ModelDeploymentStatus.SUCCESSFUL_UNDEPLOYMENT
            )
            return GetDeploymentStatusResponse(status=status)
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def GetDeploymentDetails(
        self, request: GetDeploymentDetailsRequest, context
    ) -> GetDeploymentDetailsResponse:
        try:
            result = {}
            environment_deployment_details_results = defaultdict(list)
            active_deployment_ids = self.model_uuid_to_active_deployment_ids[
                request.model_uuid
            ]
            for active_deployment in active_deployment_ids:
                deployment_details = self.get_deployment_by_id(active_deployment)
                deployment_details_proto: DeploymentDetails = (
                    self.__deployment_details_to_proto(deployment_details)
                )
                environment_deployment_details_results[
                    deployment_details_proto.environment_id
                ].append(deployment_details_proto)

            for env, deployments_list in environment_deployment_details_results.items():
                result[env] = EnvironmentDeploymentDetailsMessage(
                    deployments_details=deployments_list,
                    fallback_variation=DEFAULT_VARIATION_NAME,
                )

            return GetDeploymentDetailsResponse(
                environment_to_deployment_details=result
            )
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def UpdateDeploymentRuntimeSettings(
        self, request: UpdateDeploymentRuntimeSettingsRequest, context
    ) -> UpdateDeploymentRuntimeSettingsResponse:
        try:
            results = {}
            for deployment_id in self.model_uuid_to_active_deployment_ids[
                request.model_uuid
            ]:
                deployment_details = self.get_deployment_by_id(deployment_id)
                if request.variation_name:
                    if (
                        self.__get_variation_name(deployment_details["hosting_service"])
                        == request.variation_name
                    ):
                        results[
                            deployment_details["env"]
                        ] = EnvironmentRuntimeDeploymentSettingsResultMessage()
                else:
                    results[
                        deployment_details["env"]
                    ] = EnvironmentRuntimeDeploymentSettingsResultMessage()
            return UpdateDeploymentRuntimeSettingsResponse(
                environment_to_runtime_deployment_settings_response=results
            )
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def ApplyModelTraffic(
        self, request: ApplyModelTrafficRequest, context
    ) -> ApplyModelTrafficResponse:
        try:
            result = {}
            for env in request.environment_ids:
                if (request.model_id, env) in self.model_traffic:
                    del self.model_traffic[(request.model_id, env)].variations[:]
                    self.model_traffic[(request.model_id, env)].variations.extend(
                        request.variations
                    )
                    result[env] = EnvironmentApplyModelTrafficResponse()
                else:
                    result[env] = EnvironmentApplyModelTrafficResponse(
                        info=NO_ACTIVE_TRAFFIC
                    )
            return ApplyModelTrafficResponse(
                environment_to_apply_model_traffic_response=result
            )
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def GetModelTraffic(
        self, request: GetModelTrafficRequest, context
    ) -> GetModelTrafficResponse:
        try:
            result = {}
            for model_and_env, traffic in self.model_traffic.items():
                model_id, env = model_and_env
                if model_id != request.model_id:
                    continue

                result[env] = EnvironmentTrafficMessage(variations=traffic.variations)

            return GetModelTrafficResponse(environment_to_model_traffic=result)
        except Exception as e:
            raise_internal_grpc_error(context, e)

    @staticmethod
    def get_deployment_key(model_id: str, variation_name: str, env: str):
        return model_id, variation_name, env

    @staticmethod
    def __get_variation_name(hosting_service):
        if (
            hosting_service.kube_deployment
            and hosting_service.kube_deployment.serving_strategy
            and hosting_service.kube_deployment.serving_strategy.realtime_config
        ):
            variation_name = (
                hosting_service.kube_deployment.serving_strategy.realtime_config.traffic_config.selected_variation_name
            )
        else:
            variation_name = DEFAULT_VARIATION_NAME
        return variation_name

    def set_deployment_to_fail_on_init(self, model_id: str, build_id: str):
        self.__deployment_to_fail_on_init.add((model_id, build_id))

    def __should_fail_deployment(self, model_id: str, build_id: str) -> bool:
        return (model_id, build_id) in self.__deployment_to_fail_on_init

    def __deployment_in_progress(self, model_id: str, environment: str) -> bool:
        return (model_id, environment) in self.models_in_progress

    def __mark_deployment_in_progress(self, model_id, environment: str):
        self.models_in_progress.add((model_id, environment))

    def get_deployment_by_id(self, deployment_id: str) -> Optional[Dict]:
        for deployment_details in self.active_deployments.values():
            if deployment_details["deployment_id"] == deployment_id:
                return deployment_details

        for deployments_history_list in self.deployments_history.values():
            for deployment_details in deployments_history_list:
                if deployment_details["deployment_id"] == deployment_id:
                    return deployment_details

        return None

    def mark_deployment_successful(
        self,
        model_id: str,
        env: str,
        variation: str = DEFAULT_VARIATION_NAME,
    ):
        deployment_key = self.get_deployment_key(model_id, variation, env)
        if deployment_key in self.active_deployments:
            old_status = self.active_deployments[(model_id, variation, env)]["status"]
            if old_status != ModelDeploymentStatus.INITIATING_DEPLOYMENT:
                raise Exception(
                    f"Cannot modify deployment status from {old_status} to SUCCESSFUL"
                )
            self.models_in_progress.remove((model_id, env))
            self.active_deployments[deployment_key][
                "status"
            ] = ModelDeploymentStatus.SUCCESSFUL_DEPLOYMENT
            return True
        return False

    def __deployment_details_to_proto(self, deployment_details):
        return DeploymentDetails(
            build_id=deployment_details["build_id"],
            number_of_pods=deployment_details[
                "hosting_service"
            ].kube_deployment.deployment_size.number_of_pods,
            cpu_fraction=deployment_details[
                "hosting_service"
            ].kube_deployment.deployment_size.cpu,
            memory_amount=deployment_details[
                "hosting_service"
            ].kube_deployment.deployment_size.memory_amount,
            memory_units=deployment_details[
                "hosting_service"
            ].kube_deployment.deployment_size.memory_units,
            number_of_workers=deployment_details[
                "hosting_service"
            ].kube_deployment.advanced_deployment_options.number_of_http_server_workers,
            http_request_timeout_ms=deployment_details[
                "hosting_service"
            ].kube_deployment.advanced_deployment_options.http_request_timeout_ms,
            kube_deployment_type=deployment_details[
                "hosting_service"
            ].kube_deployment.kube_deployment_type,
            serving_strategy=deployment_details[
                "hosting_service"
            ].kube_deployment.serving_strategy,
            variation=Variation(
                name=self.__get_variation_name(deployment_details["hosting_service"])
            ),
            custom_iam_role_arn=deployment_details[
                "hosting_service"
            ].kube_deployment.advanced_deployment_options.custom_iam_role_arn,
            max_batch_size=deployment_details[
                "hosting_service"
            ].kube_deployment.advanced_deployment_options.max_batch_size,
            gpu_type=deployment_details[
                "hosting_service"
            ].kube_deployment.deployment_size.gpu_resources.gpu_type,
            gpu_amount=deployment_details[
                "hosting_service"
            ].kube_deployment.deployment_size.gpu_resources.gpu_amount,
            environment_id=deployment_details["env"],
            available_replicas=2,
        )
