from typing import Dict, List, Optional, Set

import grpc
from _qwak_proto.qwak.ecosystem.v0.ecosystem_pb2 import (
    EnvironmentDetails,
    UserContextAccountDetails,
    UserContextEnvironmentDetails,
)
from _qwak_proto.qwak.ecosystem.v0.ecosystem_runtime_service_pb2 import (
    GetAuthenticatedUserContextRequest,
    GetAuthenticatedUserContextResponse,
    GetCloudCredentialsRequest,
    GetCloudCredentialsResponse,
)
from _qwak_proto.qwak.ecosystem.v0.ecosystem_runtime_service_pb2_grpc import (
    QwakEcosystemRuntimeStub,
)
from dependency_injector.wiring import Provide
from qwak.exceptions import QwakException
from qwak.inner.di_configuration import QwakContainer


class QwakSuggestionException:
    pass


class EcosystemClient:
    """
    Used for interacting with Qwak's Ecosystem
    """

    def __init__(self, grpc_channel=Provide[QwakContainer.core_grpc_channel]):
        self._ecosystem_service = QwakEcosystemRuntimeStub(grpc_channel)

    def get_cloud_credentials(
        self, request: GetCloudCredentialsRequest
    ) -> GetCloudCredentialsResponse:
        try:
            return self._ecosystem_service.GetCloudCredentials(request)
        except grpc.RpcError as e:
            raise QwakException(
                f"Failed to get cloud credentials, error is {e.details()}"
            )

    def get_authenticated_user_context(self) -> GetAuthenticatedUserContextResponse:
        try:
            return self._ecosystem_service.GetAuthenticatedUserContext(
                GetAuthenticatedUserContextRequest()
            ).authenticated_user_context

        except grpc.RpcError as e:
            raise QwakException(f"Failed to get user context, error is {e.details()}")

    def get_account_details(self) -> UserContextAccountDetails:
        return self.get_authenticated_user_context().user.account_details

    def translate_environments_names_to_ids(
        self,
        environments_names: List[str],
    ) -> List[str]:
        names_to_selected_environments = self.get_environments_names_to_details(
            environments_names
        )

        return [
            names_to_selected_environments[env_name].id
            for env_name in environments_names
        ]

    def get_environments_names_to_details(
        self,
        environments_names: List[str],
    ) -> Dict[str, UserContextEnvironmentDetails]:
        account_details = self.get_account_details()

        if not environments_names:
            environment = account_details.environment_by_id[
                account_details.default_environment_id
            ]
            return {environment.name: environment}

        environments_names_set = set(environments_names)

        selected_environments = self.get_envs_by_names(
            account_details, environments_names_set
        )
        if len(selected_environments) != len(environments_names):
            available_environments_names = set(
                self.get_available_envs_names(account_details)
            )
            missing_envs = environments_names_set - available_environments_names

            raise QwakException(
                message=f"The following environments were not found: {','.join(missing_envs)},"
                f" Available environments are: {','.join(available_environments_names)}",
            )

        return {env.name: env for env in selected_environments}

    def get_environments(self) -> Dict[str, UserContextEnvironmentDetails]:
        account = self.get_account_details()

        return account.environment_by_id

    def get_environment_configuration(
        self, environment_name: Optional[str] = None
    ) -> EnvironmentDetails:
        account_details = self.get_account_details()
        if environment_name:
            matching_envs = self.get_envs_by_names(account_details, {environment_name})
            if not matching_envs:
                available_environments_names = self.get_available_envs_names(
                    account_details
                )
                raise QwakException(
                    f"Environment with name {environment_name} was not found. Available "
                    f"environments are: {', '.join(available_environments_names)}"
                )
            return matching_envs[0].configuration

        else:
            return account_details.environment_by_id[
                account_details.default_environment_id
            ]

    def get_environment_model_api(self, environment_name: Optional[str] = None) -> str:
        return self.get_environment_configuration(environment_name).model_api_url

    @staticmethod
    def get_available_envs_names(
        account_details: UserContextAccountDetails,
    ) -> List[str]:
        available_environments_names = list(
            map(
                lambda env: env.name,
                dict(account_details.environment_by_id).values(),
            )
        )
        return available_environments_names

    @staticmethod
    def get_envs_by_names(
        account_details: UserContextAccountDetails, environment_names: Set[str]
    ) -> List[UserContextEnvironmentDetails]:
        matching_envs = list(
            filter(
                lambda env: env.name in environment_names,
                dict(account_details.environment_by_id).values(),
            )
        )
        return matching_envs
