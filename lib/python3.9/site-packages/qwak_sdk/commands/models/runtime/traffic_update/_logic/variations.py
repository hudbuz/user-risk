from typing import Dict, List, Optional

from _qwak_proto.qwak.deployment.deployment_pb2 import (
    EnvironmentTrafficMessage,
    Variation,
)
from _qwak_proto.qwak.ecosystem.v0.ecosystem_pb2 import EnvironmentDetails
from qwak.clients.deployment.client import DeploymentManagementClient
from qwak.exceptions import QwakException

from qwak_sdk.commands.models._logic.variations import (
    validate_percentages,
    validate_shadow_variation,
)


def validate_requested_variations_against_existing_in_apply(
    model_id: str,
    requested_variations: List[Variation],
    environment_names_to_details: Dict[str, EnvironmentDetails],
    deployment_client: DeploymentManagementClient,
):
    model_traffic = deployment_client.get_model_traffic_config(model_id)
    environments_traffic = dict(model_traffic.environment_to_model_traffic)
    errors = []
    for env_name, env_details in environment_names_to_details.items():
        error = validate_requested_variation_in_envs_for_apply(
            env_details, environments_traffic, requested_variations
        )
        if error:
            errors.append(error)
    if errors:
        raise QwakException("\n".join(errors))


def validate_requested_variation_in_envs_for_apply(
    env_details: EnvironmentDetails,
    environments_traffic: Dict[str, EnvironmentTrafficMessage],
    requested_variations: List[Variation],
) -> Optional[str]:
    environment_traffic = environments_traffic.get(env_details.id)
    existing_variations = environment_traffic.variations if environment_traffic else []
    try:
        validate_variation_for_apply(
            existing_variations=existing_variations,
            requested_variations=requested_variations,
        )
    except QwakException as e:
        return e.message


def validate_variation_for_apply(
    existing_variations: List[Variation],
    requested_variations: List[Variation],
):
    requested_variations_names = set(
        map(lambda variation: variation.name, requested_variations)
    )
    existing_variations_names = set(
        map(lambda variation: variation.name, existing_variations)
    )
    ignored_variations = existing_variations_names.difference(
        requested_variations_names
    )
    if ignored_variations:
        raise QwakException(
            f"The given variation configuration does not contain the configuration for the following "
            f"variations {list(ignored_variations)}. You must include all the existing variations when updating the "
            f"model traffic {list(existing_variations_names)}"
        )

    unexpected_variations = requested_variations_names.difference(
        existing_variations_names
    )

    if unexpected_variations:
        raise QwakException(
            f"The given variation configuration contains the configuration for the following "
            f"variations {list(unexpected_variations)} which are not expected. You must include only the existing "
            f"variations when updating the model traffic {list(existing_variations_names)}"
        )

    validate_percentages(requested_variations)
    validate_shadow_variation(requested_variations)
