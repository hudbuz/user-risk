from __future__ import annotations

from typing import List

from _qwak_proto.qwak.deployment.deployment_pb2 import TrafficSpec, Variation
from qwak.exceptions import QwakException

from qwak_sdk.commands.models._logic.variations import (
    validate_percentages,
    validate_shadow_variation,
)


def get_variations_for_deploy(
    variation_name: str,
    existing_variations: List[Variation],
    requested_variations: List[Variation],
    environment_name: str,
    variation_protected_state: bool,
):
    if not existing_variations:
        existing_variations: List[Variation] = [
            Variation(
                name=variation_name,
                traffic=TrafficSpec(
                    percentage=100,
                    is_shadow=False,
                ),
                protected=variation_protected_state,
            )
        ]
    requested_variations_names = set(
        map(lambda variation: variation.name, requested_variations)
    )
    existing_variations_names = set(
        map(lambda variation: variation.name, existing_variations)
    )
    if requested_variations_names:
        if variation_name not in requested_variations_names:
            raise QwakException(
                f"Selected deployment to variation '{variation_name}' in {environment_name}, but provided the "
                f"following variations {list(requested_variations_names)}. "
                f"Please modify the variation name, or adjust the requested variations definitions"
            )
    elif variation_name not in existing_variations_names:
        raise QwakException(
            f"Selected deployment to existing variation '{variation_name}' in {environment_name}, but it is not one of "
            f"the existing variations {list(existing_variations_names)}. "
            f"Please modify the variation name, or supply the requested variations definitions with"
            f" the necessary changes"
        )
    else:
        # We are deploying to an existing variations without any change
        return existing_variations

    ignored_variations = existing_variations_names.difference(
        requested_variations_names
    )
    if ignored_variations:
        raise QwakException(
            f"The given variation configuration does not contain the configuration for the following "
            f"variations {list(ignored_variations)} in {environment_name}. You must include all the existing "
            f"variations configuration when passing the configuration"
        )

    unexpected_variations = requested_variations_names.difference(
        existing_variations_names
    )
    if variation_name in unexpected_variations:
        unexpected_variations.remove(variation_name)
    if unexpected_variations:
        raise QwakException(
            f"The given variation configuration contains the configuration for the following "
            f"variations {list(unexpected_variations)} in {environment_name} which are not expected. When deploying, "
            f"the variations configuration must contain all the existing variations, and an optional new variation."
        )

    validate_percentages(requested_variations)
    validate_shadow_variation(requested_variations)

    return requested_variations
