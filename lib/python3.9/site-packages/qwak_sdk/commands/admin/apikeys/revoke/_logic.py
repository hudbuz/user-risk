from _qwak_proto.qwak.self_service.user.v1.user_service_pb2 import RevokeApiKeyResponse
from qwak.clients.administration import SelfServiceUserClient
from qwak.clients.administration.eco_system.client import EcosystemClient


def execute_revoke_apikey(environment_name: str, user_id: str) -> bool:
    ecosystem_client = EcosystemClient()
    environment_name_to_environment_details_map = (
        ecosystem_client.get_environments_names_to_details([environment_name])
    )

    user_service = SelfServiceUserClient()

    return (
        user_service.revoke_apikey(
            user_id=user_id,
            environment_id=environment_name_to_environment_details_map.get(
                environment_name
            ).id,
        ).status
        == RevokeApiKeyResponse.Status.REVOKED
    )
