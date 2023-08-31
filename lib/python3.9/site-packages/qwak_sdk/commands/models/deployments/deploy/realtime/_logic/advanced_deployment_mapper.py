from _qwak_proto.qwak.deployment.deployment_pb2 import AdvancedDeploymentOptions
from google.protobuf.wrappers_pb2 import BoolValue

from qwak_sdk.commands.models.deployments.deploy._logic.deploy_config import (
    DeployConfig,
)


def realtime_advanced_deployment_options_from_deploy_config(
    deploy_config: DeployConfig,
) -> AdvancedDeploymentOptions:
    return AdvancedDeploymentOptions(
        number_of_http_server_workers=deploy_config.realtime.workers,
        http_request_timeout_ms=deploy_config.realtime.timeout,
        daemon_mode=BoolValue(value=deploy_config.realtime.daemon_mode),
        max_batch_size=deploy_config.realtime.max_batch_size,
        custom_iam_role_arn=deploy_config.advanced_options.iam_role_arn,
        purchase_option=deploy_config.advanced_options.purchase_option,
        deployment_process_timeout_limit=deploy_config.realtime.deployment_timeout,
    )
