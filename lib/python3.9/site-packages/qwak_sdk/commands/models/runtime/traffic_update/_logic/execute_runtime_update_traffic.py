from typing import Tuple

from qwak.clients.administration.eco_system.client import EcosystemClient
from qwak.clients.deployment.client import DeploymentManagementClient
from qwak.tools.logger.logger import get_qwak_logger

from qwak_sdk.commands.models._logic.variations import (
    create_variation_from_variation_config,
)
from qwak_sdk.commands.models.deployments.deploy._logic.deploy_config import (
    DeployConfig,
)
from qwak_sdk.commands.models.runtime.traffic_update._logic.variations import (
    validate_requested_variations_against_existing_in_apply,
)
from qwak_sdk.inner.tools.config_handler import config_handler

logger = get_qwak_logger()


def execute_runtime_traffic_update(
    model_id: str, from_file: str, environment_name: Tuple[str]
):
    deployment_client = DeploymentManagementClient()
    ecosystem_client = EcosystemClient()
    config: DeployConfig = config_handler(
        config=DeployConfig,
        from_file=from_file,
        out_conf=False,
        sections=("realtime",),
        model_id=model_id,
    )
    requested_variations = list(
        map(
            create_variation_from_variation_config,
            config.realtime.variations,
        )
    )
    environment_names_to_details = ecosystem_client.get_environments_names_to_details(
        environment_name if environment_name else config.realtime.environments
    )
    validate_requested_variations_against_existing_in_apply(
        model_id,
        requested_variations,
        environment_names_to_details,
        deployment_client,
    )
    environment_ids = [env.id for env in environment_names_to_details.values()]

    deployment_client.apply_model_traffic_config(
        model_id=model_id,
        requested_variations=requested_variations,
        environment_ids=environment_ids,
    )
