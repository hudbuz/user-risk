from _qwak_proto.qwak.deployment.deployment_pb2 import AdvancedDeploymentOptions

from qwak_sdk.commands.models.deployments.deploy._logic.deploy_config import (
    DeployConfig,
)


def batch_advanced_deployment_options_from_deploy_config(
    deploy_config: DeployConfig,
) -> AdvancedDeploymentOptions:
    return AdvancedDeploymentOptions(
        custom_iam_role_arn=deploy_config.advanced_options.iam_role_arn,
        purchase_option=deploy_config.advanced_options.purchase_option,
    )
