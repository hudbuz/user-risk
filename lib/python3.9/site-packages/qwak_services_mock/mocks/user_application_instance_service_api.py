from _qwak_proto.qwak.admiral.user_application_instance.v0.user_application_instance_pb2 import (
    Description,
    Identifier,
    Metadata,
    SpecDescription,
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
    UserApplicationInstanceServiceServicer,
)
from qwak_services_mock.mocks.utils.exception_handlers import raise_internal_grpc_error


class UserApplicationInstanceServiceApiMock(UserApplicationInstanceServiceServicer):
    def __init__(self):
        self.revisions = {}
        super(UserApplicationInstanceServiceApiMock, self).__init__()

    @staticmethod
    def _make_key_from_identifier(identifier: Identifier):
        return (
            identifier.name
            + "|"
            + str(identifier.type)
            + "|"
            + identifier.account_id
            + "|"
            + identifier.environment_id
        )

    @staticmethod
    def _get_identifier_from_key(key: str) -> Identifier:
        splitted_key = key.split("|")
        return Identifier(
            name=splitted_key[0],
            type=int(splitted_key[1]),
            account_id=splitted_key[2],
            environment_id=splitted_key[3],
        )

    def CreateUserApplicationInstance(
        self, request: CreateUserApplicationInstanceRequest, context
    ):
        try:
            key = self._make_key_from_identifier(request.create_options.identifier)
            spec = self.revisions.get(key)
            if spec is not None:
                raise Exception("User application already exists")
            self.revisions[key] = request.create_options.spec
            return CreateUserApplicationInstanceResponse()
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def UpdateUserApplicationInstanceSpec(
        self, request: UpdateUserApplicationInstanceSpecRequest, context
    ):
        try:
            key = self._make_key_from_identifier(request.update_options.identifier)
            revision = self.revisions.get(key)
            if revision is None:
                raise Exception("Get Update request but no revision was found")
            self.revisions[key] = request.update_options.spec
            return UpdateUserApplicationInstanceSpecResponse()
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def GetUserApplicationInstance(
        self, request: GetUserApplicationInstanceRequest, context
    ):
        try:
            key = self._make_key_from_identifier(request.identifier)
            spec = self.revisions.get(key)
            if spec is None:
                return GetUserApplicationInstanceResponse()
            description = Description(
                metadata=Metadata(
                    identifier=self._get_identifier_from_key(key=key),
                ),
                spec_description=(SpecDescription(spec=spec)),
            )
            return GetUserApplicationInstanceResponse(description=description)
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def DeleteUserApplicationInstance(
        self, request: DeleteUserApplicationInstanceRequest, context
    ):
        try:
            key = self._make_key_from_identifier(request.identifier)
            spec = self.revisions[key]
            if spec is None:
                raise Exception(f"Could not find revision by the identifier key: {key}")
            self.revisions[key] = None
            return DeleteUserApplicationInstanceResponse()
        except Exception as e:
            raise_internal_grpc_error(context, e)
