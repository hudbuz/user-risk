from typing import Dict

from _qwak_proto.qwak.deployment.deployment_pb2 import ModelDeploymentStatus
from _qwak_proto.qwak.deployment.deployment_service_pb2 import DeployModelResponse
from _qwak_proto.qwak.ecosystem.v0.ecosystem_pb2 import UserContextEnvironmentDetails
from qwak.clients.administration.eco_system.client import EcosystemClient
from qwak.exceptions import QwakException
from qwak.tools.logger.logger import get_qwak_logger
from tabulate import tabulate

from qwak_sdk.commands.models.deployments.deploy._logic.base_deploy_executor import (
    FAILED_STATUS,
    BaseDeployExecutor,
)
from qwak_sdk.tools.utils import qwak_spinner

logger = get_qwak_logger()


def __client_deployment_remote(
    deploy: BaseDeployExecutor, sync: bool, output_multiple: bool = False
):
    try:
        deploy_response: DeployModelResponse = deploy.deploy()
        deploy_status, deploy_id = (
            ModelDeploymentStatus.Name(deploy_response.status),
            deploy_response.deployment_named_id,
        )

        if deploy_response.environment_to_deployment_result:
            account_environments = EcosystemClient().get_environments()
            output_deployments_response(account_environments, deploy, deploy_response)
            __check_failed_deployments(account_environments, deploy_response)

            if sync:
                deployment_id_to_env = {}
                for (
                    env_id,
                    env_deploy_response,
                ) in deploy_response.environment_to_deployment_result.items():
                    deployment_id_to_env[
                        env_deploy_response.deployment_named_id
                    ] = account_environments.get(env_id).name

                __sync_deployments(deploy, deployment_id_to_env)

            if output_multiple:
                return deploy_response.environment_to_deployment_result
        else:
            if "FAILED" in deploy_status:
                raise QwakException(deploy_response.info)

            if sync:
                __sync_deployment(deploy, deploy_id)

        if not sync:
            logger.info(
                "Deployment initiated successfully, Use --sync to wait for deployments to be ready."
            )
            return deploy_id, deploy_status
    except Exception as e:
        logger.error(f"Deployment failed, Error: {e}")
        raise e


def __check_failed_deployments(
    account_environments: Dict[str, UserContextEnvironmentDetails],
    deploy_response: DeployModelResponse,
):
    failed_deployments = {}
    for (
        env_id,
        env_deploy_response,
    ) in deploy_response.environment_to_deployment_result.items():
        deploy_status = ModelDeploymentStatus.Name(env_deploy_response.status)
        if "FAILED" in deploy_status:
            env_config = account_environments.get(env_id)
            failed_deployments[
                env_config.name if env_config else env_id
            ] = env_deploy_response.info
    if failed_deployments:
        failed_deployments_message = "\n".join(
            [
                f"Environment {env_name}: {reason}"
                for env_name, reason in failed_deployments.items()
            ]
        )
        raise QwakException(
            f"The following deployments failed:\n {failed_deployments_message}"
        )


def output_deployments_response(
    account_environments: Dict[str, UserContextEnvironmentDetails],
    deploy: BaseDeployExecutor,
    deploy_response: DeployModelResponse,
):
    for (
        env_id,
        env_deploy_response,
    ) in deploy_response.environment_to_deployment_result.items():
        env_config = account_environments.get(env_id)
        if env_deploy_response.status == ModelDeploymentStatus.INITIATING_DEPLOYMENT:
            logger.info(
                tabulate(
                    tabular_data=[
                        ["Environment", env_config.name],
                        ["Model ID", deploy.config.model_id],
                        ["Build ID", deploy.config.build_id],
                        ["Deployment ID", env_deploy_response.deployment_named_id],
                    ],
                    tablefmt="fancy_grid",
                )
            )


def __sync_deployments(
    deploy: BaseDeployExecutor, deployment_id_to_env: Dict[str, str]
):
    with qwak_spinner(
        begin_text=f"Deploy - model: {deploy.config.model_id}, build: {deploy.config.build_id} "
        f"Environments: {', '.join(deploy.config.realtime.environments)}",
        end_text="Successful deployments",
        print_callback=print,
    ):
        deployment_id_to_status = deploy.poll_until_complete_multiple(
            list(deployment_id_to_env.keys()), check_every_n_seconds=5
        )
        failed_deployments = []
        for deployment_id, status in deployment_id_to_status.items():
            if status in FAILED_STATUS:
                failed_deployments.append(
                    f"Environment {deployment_id_to_env.get(deployment_id)} "
                    f"failed with status: {status}"
                )
        if failed_deployments:
            raise QwakException("\n".join(failed_deployments))


def __sync_deployment(deploy: BaseDeployExecutor, deploy_name: str):
    with qwak_spinner(
        begin_text=f"Deploy - model: {deploy.config.model_id}, build: {deploy.config.build_id}",
        end_text="Successful deployments",
        print_callback=print,
    ):
        deployment_status = deploy.poll_until_complete(
            deploy_name, check_every_n_seconds=5
        )
        if deployment_status in FAILED_STATUS:
            raise QwakException(deployment_status)


def client_deployment(
    deploy: BaseDeployExecutor, sync: bool, output_multiple: bool = False
):
    return __client_deployment_remote(deploy, sync, output_multiple)
