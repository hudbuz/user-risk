from typing import Dict, List, Set

from _qwak_proto.qwak.audience.v1.audience_pb2 import AudienceRoutesEntry
from _qwak_proto.qwak.deployment.deployment_pb2 import (
    DeploymentHostingServiceType,
    EnvironmentDeploymentDetailsMessage,
    EnvironmentUndeploymentMessage,
    ModelDeploymentStatus,
    TrafficConfig,
    Variation,
)
from _qwak_proto.qwak.ecosystem.v0.ecosystem_pb2 import EnvironmentDetails
from qwak.clients.administration.eco_system.client import EcosystemClient
from qwak.clients.deployment.client import DeploymentManagementClient
from qwak.exceptions import QwakException
from qwak.tools.logger.logger import get_qwak_logger

from qwak_sdk.commands.models._logic.variations import (
    create_variation_from_variation_config,
)
from qwak_sdk.commands.models.deployments.deploy._logic.deploy_config import (
    DeployConfig,
)
from qwak_sdk.commands.models.deployments.undeploy._logic.variations import (
    validate_variations_for_undeploy,
)

NO_DEPLOYED_VARIATIONS_ERROR_MSG = (
    "There are currently no deployed variations for model {model_id} in {env_name}"
)

logger = get_qwak_logger()


def get_deployed_variation_name(existing_variations_names: Set[str]) -> str:
    return list(existing_variations_names)[0]


def get_environment_undeploy_message(
    audiences: List[AudienceRoutesEntry],
    env_name: str,
    existing_variations_names: Set[str],
    fallback_variation: str,
    model_id: str,
    model_uuid: str,
    requested_variations: List[Variation],
    variation_name: str,
):
    if not variation_name and len(existing_variations_names) == 1:
        variation_name = get_deployed_variation_name(
            existing_variations_names=existing_variations_names
        )

    if not audiences:
        validate_variations_for_undeploy(
            variation_name,
            existing_variations_names,
            requested_variations,
            env_name,
        )
    return EnvironmentUndeploymentMessage(
        model_id=model_id,
        model_uuid=model_uuid,
        hosting_service_type=DeploymentHostingServiceType.KUBE_DEPLOYMENT,
        traffic_config=TrafficConfig(
            selected_variation_name=variation_name,
            variations=requested_variations,
            audience_routes_entries=audiences,
            fallback_variation=fallback_variation,
        ),
    )


def get_env_to_undeploy_message(
    audiences: List[AudienceRoutesEntry],
    model_uuid: str,
    env_id_to_deployment_details: Dict[str, EnvironmentDeploymentDetailsMessage],
    env_name_to_env_details: Dict[str, EnvironmentDetails],
    model_id: str,
    requested_variations: List[Variation],
    variation_name: str,
    fallback_variation: str,
) -> Dict[str, EnvironmentUndeploymentMessage]:
    env_undeployment_requests = dict()
    errors = []
    for env_name, env_details in env_name_to_env_details.items():
        env_deployments_details_message = env_id_to_deployment_details.get(
            env_details.id
        )
        if not env_deployments_details_message:
            errors.append(
                NO_DEPLOYED_VARIATIONS_ERROR_MSG.format(
                    model_id=model_id, env_name=env_name
                )
            )
            continue

        env_deployments_details = env_deployments_details_message.deployments_details
        existing_variations_names = {
            deployment.variation.name for deployment in env_deployments_details
        }
        try:
            env_undeployment_requests[
                env_details.id
            ] = get_environment_undeploy_message(
                audiences,
                env_name,
                existing_variations_names,
                fallback_variation,
                model_id,
                model_uuid,
                requested_variations,
                variation_name,
            )
        except QwakException as e:
            errors.append(e.message)
    if errors:
        raise QwakException("\n".join(errors))
    return env_undeployment_requests


def undeploy(
    model_id: str,
    config: DeployConfig,
    model_uuid: str = "",
):
    deployment_client = DeploymentManagementClient()
    ecosystem_client = EcosystemClient()
    audiences: List[AudienceRoutesEntry] = [
        audience.to_audience_route_entry(index)
        for index, audience in enumerate(config.realtime.audiences)
    ]
    requested_variations = list(
        map(
            create_variation_from_variation_config,
            config.realtime.variations if config.realtime else [],
        )
    )

    if not model_uuid:
        raise QwakException("missing argument model uuid")

    environments_names = config.realtime.environments if config.realtime else []

    deployment_details = deployment_client.get_deployment_details(model_id, model_uuid)
    env_id_to_deployment_details = dict(
        deployment_details.environment_to_deployment_details
    )
    env_name_to_env_details = ecosystem_client.get_environments_names_to_details(
        environments_names
    )

    env_undeployment_requests = get_env_to_undeploy_message(
        audiences,
        model_uuid,
        env_id_to_deployment_details,
        env_name_to_env_details,
        model_id,
        requested_variations,
        config.realtime.variation_name,
        config.realtime.fallback_variation,
    )

    undeployment_response = deployment_client.undeploy_model(
        model_id=model_id,
        model_uuid=model_uuid,
        env_undeployment_requests=env_undeployment_requests,
    )

    logger.info(
        f"Current status is {ModelDeploymentStatus.Name(undeployment_response.status)}."
    )

    return undeployment_response
