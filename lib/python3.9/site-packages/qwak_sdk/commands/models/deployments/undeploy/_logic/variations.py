from __future__ import annotations

from typing import List, Set

from _qwak_proto.qwak.deployment.deployment_pb2 import Variation
from qwak.exceptions import QwakException

from qwak_sdk.commands.models.deployments.deploy._logic.variations import (
    validate_percentages,
    validate_shadow_variation,
)


def validate_variations_for_undeploy(
    variation_name: str,
    existing_variations_names: Set[str],
    requested_variations: List[Variation],
    environment_name: str,
):
    if variation_name not in existing_variations_names:
        if not variation_name:
            raise QwakException(
                f"No selected variation. You must specify a variation since there is more than one deployed in "
                f"{environment_name}. Currently {list(existing_variations_names)} are deployed"
            )
        raise QwakException(
            f"The selected variation {variation_name} is not deployed in {environment_name}. "
            f"Only {list(existing_variations_names)} are deployed"
        )

    requested_variations_names = set(
        map(lambda variation: variation.name, requested_variations)
    )
    if variation_name in requested_variations_names:
        raise QwakException(
            f"Selected undeploy to variation '{variation_name}', but it is part of "
            f"the requested variations {list(requested_variations_names)}. "
            f"Please modify the variation name, or supply the requested variations definitions with"
            f" the necessary changes"
        )
    if variation_name not in existing_variations_names:
        raise QwakException(
            f"Selected undeploy to variation {variation_name} but it is not one of "
            f"the existing variations {list(existing_variations_names)}. "
            f"Please modify the variation name."
        )

    ignored_variations = existing_variations_names.difference(
        requested_variations_names
    )
    if variation_name in ignored_variations:
        ignored_variations.remove(variation_name)

    if len(existing_variations_names) != 2 and ignored_variations:
        raise QwakException(
            f"The given variation configuration does not contain the configuration for the following "
            f"variations {list(ignored_variations)}. You must include all the variations that will exist after "
            f"undeploy in the configuration"
        )

    unexpected_variations = requested_variations_names.difference(
        existing_variations_names
    )

    if unexpected_variations:
        raise QwakException(
            f"The given variation configuration contains the configuration for the following "
            f"variations {list(unexpected_variations)} which are not expected. When undeploying, the variations "
            f"configuration must contain all expected remaining variations (from the existing ones)."
        )

    if requested_variations:
        validate_percentages(requested_variations)
        validate_shadow_variation(requested_variations)
