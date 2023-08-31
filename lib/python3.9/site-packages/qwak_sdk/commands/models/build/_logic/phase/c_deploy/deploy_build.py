from __future__ import annotations

from typing import List

from _qwak_proto.qwak.instance_template.instance_template_pb2 import (
    InstanceTemplateSpec,
    InstanceType,
)

from qwak_sdk.commands.models.build._logic.client_logs.messages import (
    FAILED_DEPLOY_BUILD_SUGGESTION,
)
from qwak_sdk.commands.models.build._logic.interface.step_inteface import Step
from qwak_sdk.commands.models.deployments.deploy.realtime.ui import deploy_realtime
from qwak_sdk.exceptions.qwak_deploy_new_build_failed import (
    QwakDeployNewBuildFailedException,
)


class DeployBuildStep(Step):
    DEPLOY_FAILURE_EXCEPTION_MESSAGE = "Deploying the build failed due to {e}"

    def description(self) -> str:
        return "Deploying Build"

    def execute(self) -> None:
        self.notifier.info(f"Deploying build {self.context.build_id}")
        try:
            if self.config.deployment_instance:
                template_id = self.config.deployment_instance
            else:
                template_id = self.get_smallest_deployment_template_id()
            deploy_config = {
                "build_id": self.context.build_id,
                "model_id": self.context.model_id,
                "instance": template_id,
            }
            deploy_realtime(from_file=None, out_conf=False, sync=False, **deploy_config)
            self.notifier.info(f"Finished deploying build {self.context.build_id}")
        except Exception as e:
            raise QwakDeployNewBuildFailedException(
                message=self.DEPLOY_FAILURE_EXCEPTION_MESSAGE.format(e=e),
                suggestion=FAILED_DEPLOY_BUILD_SUGGESTION.format(
                    build_id=self.context.build_id,
                    model_id=self.context.model_id,
                    project_uuid=self.context.project_uuid,
                ),
            )

    def get_smallest_deployment_template_id(self):
        instances: List[
            InstanceTemplateSpec
        ] = self.context.client_instance_template.list_instance_templates()
        return list(
            filter(
                lambda template: template.order == 1
                and template.instance_type == InstanceType.INSTANCE_TYPE_CPU,
                instances,
            )
        )[0].id
