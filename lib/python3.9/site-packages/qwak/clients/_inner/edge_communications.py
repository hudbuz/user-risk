from typing import Optional

from qwak.clients.administration.eco_system.client import EcosystemClient
from qwak.exceptions import QwakException


def get_endpoint_url(
    endpoint_url: Optional[str] = None, environment_id: Optional[str] = None
) -> str:
    if not endpoint_url:
        user_context = EcosystemClient().get_authenticated_user_context().user
        if not environment_id:
            environment_id = user_context.account_details.default_environment_id

        if environment_id not in user_context.account_details.environment_by_id:
            raise QwakException(
                f"Configuration for environment '{environment_id}' was not found"
            )

        endpoint_url = user_context.account_details.environment_by_id[
            environment_id
        ].configuration.edge_services_url

    return endpoint_url
