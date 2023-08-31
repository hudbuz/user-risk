from typing import Optional

import grpc
from _qwak_proto.qwak.admiral.user_application_instance.v0.user_application_instance_pb2 import (
    CreateOptions,
    Identifier,
    RevisionRestriction,
    UpdateSpecOptions,
)
from _qwak_proto.qwak.admiral.user_application_instance.v0.user_application_instance_service_pb2 import (
    CreateUserApplicationInstanceRequest,
    CreateUserApplicationInstanceResponse,
    DeleteUserApplicationInstanceRequest,
    DeleteUserApplicationInstanceResponse,
    GetUserApplicationInstanceRequest,
    GetUserApplicationInstanceResponse,
    UpdateUserApplicationInstanceSpecRequest,
    UpdateUserApplicationInstanceSpecResponse,
)
from _qwak_proto.qwak.admiral.user_application_instance.v0.user_application_instance_service_pb2_grpc import (
    UserApplicationInstanceServiceStub,
)
from _qwak_proto.qwak.user_application.v0.user_application_pb2 import Spec, Type
from dependency_injector.wiring import Provide
from qwak.exceptions import QwakException, QwakNotFoundException
from qwak.inner.di_configuration import QwakContainer


class UserApplicationInstanceClient:
    """
    Used for interacting with Feature Registry endpoints
    """

    def __init__(self, grpc_channel=Provide[QwakContainer.core_grpc_channel]):
        self._user_application_instance_service = UserApplicationInstanceServiceStub(
            grpc_channel
        )

    def create_user_application_instance(
        self,
        kind_sem_version: Optional[str],
        account_id: str,
        environment_id: str,
        name: str,
        type: Type,
        spec: Spec,
    ) -> CreateUserApplicationInstanceResponse:
        """
        Create user application instance
        """
        try:
            identifier = Identifier(
                account_id=account_id,
                environment_id=environment_id,
                name=name,
                type=type,
            )

            create_options = CreateOptions(
                identifier=identifier, spec=spec, kind_sem_version=kind_sem_version
            )
            create_user_application_instance_request = (
                CreateUserApplicationInstanceRequest(create_options=create_options)
            )

            return (
                self._user_application_instance_service.CreateUserApplicationInstance(
                    create_user_application_instance_request
                )
            )
        except grpc.RpcError as e:
            raise QwakException(
                f"Failed to create user application instance, error is {repr(e)}"
            )

    def update_user_application_instance(
        self,
        current_spec_revision: Optional[int],
        account_id: str,
        environment_id: str,
        name: str,
        type: Type,
        spec: Spec,
    ) -> UpdateUserApplicationInstanceSpecResponse:
        """
        Update user application instance
        """
        try:
            identifier = Identifier(
                account_id=account_id,
                environment_id=environment_id,
                name=name,
                type=type,
            )

            revision_restriction = RevisionRestriction(
                current_spec_revision=current_spec_revision
            )
            update_options = UpdateSpecOptions(
                identifier=identifier,
                spec=spec,
                revision_restriction=revision_restriction,
            )
            update_user_application_instance_request = (
                UpdateUserApplicationInstanceSpecRequest(update_options=update_options)
            )

            return self._user_application_instance_service.UpdateUserApplicationInstanceSpec(
                update_user_application_instance_request
            )
        except grpc.RpcError as e:
            raise QwakException(
                f"Failed to update user application instance, error is {repr(e)}"
            )

    def delete_user_application_instance(
        self,
        account_id: str,
        environment_id: str,
        name: str,
        type: Type,
    ) -> DeleteUserApplicationInstanceResponse:
        """
        Delete user application instance
        """
        try:
            identifier = Identifier(
                account_id=account_id,
                environment_id=environment_id,
                name=name,
                type=type,
            )
            delete_user_application_instance_request = (
                DeleteUserApplicationInstanceRequest(identifier=identifier)
            )
            return (
                self._user_application_instance_service.DeleteUserApplicationInstance(
                    delete_user_application_instance_request
                )
            )
        except grpc.RpcError as e:
            raise QwakException(
                f"Failed to delete user application instance, error is {repr(e)}"
            )

    def get_user_application_instance(
        self,
        account_id: str,
        environment_id: str,
        name: str,
        type: Type,
    ) -> GetUserApplicationInstanceResponse:
        """
        Get user application instance
        """
        try:
            identifier = Identifier(
                account_id=account_id,
                environment_id=environment_id,
                name=name,
                type=type,
            )
            get_user_application_request = GetUserApplicationInstanceRequest(
                identifier=identifier
            )
            return self._user_application_instance_service.GetUserApplicationInstance(
                get_user_application_request
            )
        except grpc.RpcError as e:
            if e.args[0].code == grpc.StatusCode.NOT_FOUND:
                raise QwakNotFoundException(
                    f"User application instance  Resource was not found, error is {repr(e)}"
                )
            raise QwakException(
                f"Failed to get user application instance, error is {repr(e)}"
            )
