from __future__ import annotations

from typing import Dict, List

import grpc
from _qwak_proto.qwak.deployment.deployment_pb2 import (
    DeploymentHostingServiceType,
    EnvironmentDeploymentMessage,
    EnvironmentUndeploymentMessage,
    RuntimeDeploymentSettings,
    Variation,
)
from _qwak_proto.qwak.deployment.deployment_service_pb2 import (
    ApplyModelTrafficRequest,
    DeployModelRequest,
    GetDeploymentDetailsRequest,
    GetDeploymentDetailsResponse,
    GetDeploymentStatusRequest,
    GetDeploymentStatusResponse,
    GetModelTrafficRequest,
    GetModelTrafficResponse,
    UndeployModelRequest,
    UpdateDeploymentRuntimeSettingsRequest,
    UpdateDeploymentRuntimeSettingsResponse,
)
from _qwak_proto.qwak.deployment.deployment_service_pb2_grpc import (
    DeploymentManagementServiceStub,
)
from dependency_injector.wiring import Provide
from qwak.exceptions import QwakException
from qwak.inner.di_configuration import QwakContainer


class DeploymentManagementClient:
    def __init__(self, grpc_channel=Provide[QwakContainer.core_grpc_channel]):
        self.deployment_management = DeploymentManagementServiceStub(grpc_channel)

    def deploy_model(
        self,
        model_id: str,
        build_id: str,
        env_deployment_messages: Dict[str, EnvironmentDeploymentMessage],
    ):
        try:
            request = DeployModelRequest(
                model_id=model_id,
                build_id=build_id,
                hosting_service=list(env_deployment_messages.values())[
                    0
                ].hosting_service,
                environment_to_deployment=env_deployment_messages,
            )

            deployment_result = self.deployment_management.DeployModel(
                request,
                timeout=40,
            )

            return deployment_result
        except grpc.RpcError as e:
            raise QwakException(f"Failed to deploy model, error is {e}")

    def get_deployment_status(
        self, deployment_named_id: str
    ) -> GetDeploymentStatusResponse:
        try:
            return self.deployment_management.GetDeploymentStatus(
                GetDeploymentStatusRequest(
                    deployment_named_id=deployment_named_id,
                    hosting_service_type=DeploymentHostingServiceType.KUBE_DEPLOYMENT,
                )
            )
        except grpc.RpcError as e:
            raise QwakException(f"Failed to get model status, error is {e.details()}")

    def get_deployment_details(
        self, model_id: str, model_uuid: str, **kwargs
    ) -> GetDeploymentDetailsResponse:
        try:
            _model_uuid = model_uuid if model_uuid else kwargs.get("branch_id")
            if not _model_uuid:
                raise QwakException("missing argument model uuid or branch id.")

            return self.deployment_management.GetDeploymentDetails(
                GetDeploymentDetailsRequest(
                    model_id=model_id,
                    model_uuid=_model_uuid,
                )
            )
        except grpc.RpcError as e:
            raise QwakException(f"Failed to get model status, error is {e.details()}")

    def undeploy_model(
        self,
        model_id: str,
        model_uuid: str,
        env_undeployment_requests: Dict[str, EnvironmentUndeploymentMessage],
    ):
        try:
            undeployment_result = self.deployment_management.UndeployModel(
                UndeployModelRequest(
                    model_id=model_id,
                    model_uuid=model_uuid,
                    hosting_service_type=DeploymentHostingServiceType.KUBE_DEPLOYMENT,
                    environment_to_undeployment=env_undeployment_requests,
                )
            )

            return undeployment_result
        except grpc.RpcError as e:
            raise QwakException(f"Failed to undeploy model, error is {e}")

    def update_runtime_configurations(
        self, model_id: str, model_uuid: str, log_level: str
    ) -> UpdateDeploymentRuntimeSettingsResponse:
        try:
            return self.deployment_management.UpdateDeploymentRuntimeSettings(
                UpdateDeploymentRuntimeSettingsRequest(
                    model_id=model_id,
                    model_uuid=model_uuid,
                    deployment_settings=RuntimeDeploymentSettings(
                        root_logger_level=RuntimeDeploymentSettings.LogLevel.Value(
                            log_level
                        )
                    ),
                )
            )
        except grpc.RpcError as e:
            raise QwakException(f"Failed to update runtime settings, error is: {e}")

    def get_model_traffic_config(self, model_id: str) -> GetModelTrafficResponse:
        try:
            return self.deployment_management.GetModelTraffic(
                GetModelTrafficRequest(
                    model_id=model_id,
                )
            )
        except grpc.RpcError as e:
            raise QwakException(f"Failed to get model traffic, error is {e.details()}")

    def get_deployed_variations_to_build_id(
        self, model_id: str, model_uuid: str
    ) -> Dict[str, str]:
        deployment_details = self.get_deployment_details(
            model_id=model_id, model_uuid=model_uuid
        )
        if not deployment_details:
            raise QwakException(
                f"There are currently no deployed variations for model {model_id}"
            )

        return {
            deployment.variation.name: deployment.build_id
            for deployment in deployment_details.current_deployments_details
        }

    def apply_model_traffic_config(
        self,
        model_id: str,
        requested_variations: List[Variation],
        environment_ids: List[str],
    ) -> GetDeploymentStatusResponse:
        try:
            return self.deployment_management.ApplyModelTraffic(
                ApplyModelTrafficRequest(
                    model_id=model_id,
                    variations=requested_variations,
                    environment_ids=environment_ids,
                )
            )
        except grpc.RpcError as e:
            raise QwakException(
                f"Failed to apply model traffic, error is {e.details()}"
            )
