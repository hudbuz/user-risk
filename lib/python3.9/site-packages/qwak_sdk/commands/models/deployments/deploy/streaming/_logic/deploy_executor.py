from _qwak_proto.qwak.deployment.deployment_pb2 import KubeDeploymentType
from _qwak_proto.qwak.deployment.deployment_service_pb2 import DeployModelResponse

from qwak_sdk.commands.models.deployments.deploy._logic.base_deploy_executor import (
    BaseDeployExecutor,
)
from qwak_sdk.commands.models.deployments.deploy._logic.deployment_message_helpers import (
    get_env_to_deployment_message,
)


class StreamDeployExecutor(BaseDeployExecutor):
    def deploy(self) -> DeployModelResponse:
        env_deployment_messages = get_env_to_deployment_message(
            self.config,
            KubeDeploymentType.STREAM,
            self.deploy_client,
            self.ecosystem_client,
            self.instance_template_client,
        )
        return self.deploy_client.deploy_model(
            model_id=self.config.model_id,
            build_id=self.config.build_id,
            env_deployment_messages=env_deployment_messages,
        )
