from __future__ import annotations

from typing import List

from _qwak_proto.qwak.deployment.deployment_pb2 import TrafficSpec, Variation
from qwak.exceptions import QwakException

from qwak_sdk.commands.models.deployments.deploy._logic.deploy_config import (
    DeployConfig,
)


def validate_percentages(variations: List[Variation]):
    for variation in variations:
        if not (0 <= variation.traffic.percentage <= 100):
            raise QwakException(
                f"The variation '{variation.name}' contains invalid value '{variation.traffic.percentage}'. "
                f"Value must be between 0 and 100."
            )

    non_shadow_variations = list(filter(lambda v: not v.traffic.is_shadow, variations))
    non_shadow_variations_percentage_sum = sum(
        map(lambda v: v.traffic.percentage, non_shadow_variations)
    )
    if non_shadow_variations and non_shadow_variations_percentage_sum != 100:
        raise QwakException(
            "The variations do not sum to 100 percent. Please go over the variations config file."
        )


def validate_shadow_variation(variations: List[Variation]):
    shadow_variations = list(filter(lambda v: v.traffic.is_shadow, variations))
    if len(shadow_variations) > 1:
        shadow_variations_names = list(map(lambda v: v.name, shadow_variations))
        raise QwakException(
            f"The variations contain more than one shadow variation {shadow_variations_names}. "
            f"Please go over the variations config file."
        )


def create_variation_from_variation_config(
    variation_def: DeployConfig.Realtime.VariationConfig,
) -> Variation:
    try:
        return Variation(
            name=variation_def.name,
            traffic=TrafficSpec(
                percentage=variation_def.traffic.percentage,
                is_shadow=variation_def.traffic.shadow,
            ),
        )
    except Exception:
        raise QwakException(
            f"Could not parse variation {variation_def}. Please check you variation configuration file."
        )
