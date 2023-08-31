from abc import ABC, abstractmethod
from time import sleep
from typing import List

from _qwak_proto.qwak.deployment.deployment_pb2 import ModelDeploymentStatus
from _qwak_proto.qwak.deployment.deployment_service_pb2 import DeployModelResponse
from qwak.clients.administration.eco_system.client import EcosystemClient
from qwak.clients.deployment.client import DeploymentManagementClient
from qwak.clients.instance_template.client import InstanceTemplateManagementClient
from qwak.inner.di_configuration import UserAccountConfiguration

from qwak_sdk.commands.models.deployments.deploy._logic.deploy_config import (
    DeployConfig,
)

FAILED_STATUS = [
    "FAILED_DEPLOYMENT",
    "FAILED_INITIATING_DEPLOYMENT",
]
SUCCESSFUL_STATUS = ["SUCCESSFUL_DEPLOYMENT"]
END_STATUSES = SUCCESSFUL_STATUS + FAILED_STATUS


class BaseDeployExecutor(ABC):
    def __init__(self, config: DeployConfig):
        self.config = config
        self.deploy_client = DeploymentManagementClient()
        self.ecosystem_client = EcosystemClient()
        self.user_config = UserAccountConfiguration().get_user_config()
        self.instance_template_client = InstanceTemplateManagementClient()

    def poll_until_complete(self, deployment_id: str, check_every_n_seconds: int):
        def deployment_status():
            status_response = self.deploy_client.get_deployment_status(
                deployment_named_id=deployment_id
            )
            if ModelDeploymentStatus.Name(status_response.status) in END_STATUSES:
                return ModelDeploymentStatus.Name(status_response.status)

        status = deployment_status()
        while status not in END_STATUSES:
            sleep(check_every_n_seconds)
            status = deployment_status()
        return status

    def poll_until_complete_multiple(
        self, deployment_ids: List[str], check_every_n_seconds: int
    ):
        def deployment_status(deployment_id: str):
            status_response = self.deploy_client.get_deployment_status(
                deployment_named_id=deployment_id
            )
            if ModelDeploymentStatus.Name(status_response.status) in END_STATUSES:
                return ModelDeploymentStatus.Name(status_response.status)

        statuses = [
            deployment_status(deployment_id) for deployment_id in deployment_ids
        ]
        while all(status not in END_STATUSES for status in statuses):
            sleep(check_every_n_seconds)
            statuses = [
                deployment_status(deployment_id) for deployment_id in deployment_ids
            ]
        return dict(zip(deployment_ids, statuses))

    @abstractmethod
    def deploy(self) -> DeployModelResponse:
        pass
