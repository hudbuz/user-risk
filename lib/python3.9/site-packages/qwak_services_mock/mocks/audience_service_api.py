import uuid

from _qwak_proto.qwak.audience.v1.audience_api_pb2 import (
    CreateAudienceRequest,
    CreateAudienceResponse,
    DeleteAudienceRequest,
    DeleteAudienceResponse,
    GetAudienceRequest,
    GetAudienceResponse,
    ListAudienceRequest,
    ListAudienceResponse,
    SyncAudiencesRequest,
    SyncAudiencesResponse,
    UpdateAudienceRequest,
    UpdateAudienceResponse,
)
from _qwak_proto.qwak.audience.v1.audience_api_pb2_grpc import AudienceAPIServicer
from _qwak_proto.qwak.audience.v1.audience_pb2 import AudienceEntry
from qwak_services_mock.mocks.utils.exception_handlers import raise_internal_grpc_error


class AudienceServiceApiMock(AudienceAPIServicer):
    def __init__(self):
        super(AudienceServiceApiMock, self).__init__()
        self.audiences = dict()

    def CreateAudience(
        self, request: CreateAudienceRequest, context
    ) -> CreateAudienceResponse:
        try:
            audience_id = str(uuid.uuid4())
            self.audiences[audience_id] = request.audience
            return CreateAudienceResponse(audience_id=audience_id)
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def UpdateAudience(
        self, request: UpdateAudienceRequest, context
    ) -> UpdateAudienceResponse:
        try:
            self.audiences[request.id] = request.audience
            return UpdateAudienceResponse()
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def GetAudience(self, request: GetAudienceRequest, context) -> GetAudienceResponse:
        try:
            return GetAudienceResponse(audience=self.audiences[request.audience_id])
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def ListAudience(
        self, request: ListAudienceRequest, context
    ) -> ListAudienceResponse:
        try:
            return ListAudienceResponse(
                audience_entries=[
                    AudienceEntry(id=audience_id, audience=audience)
                    for audience_id, audience in self.audiences.items()
                ]
            )
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def DeleteAudience(
        self, request: DeleteAudienceRequest, context
    ) -> DeleteAudienceResponse:
        try:
            del self.audiences[request.audience_id]
            return DeleteAudienceResponse()
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def SyncAudiences(
        self, request: SyncAudiencesRequest, context
    ) -> SyncAudiencesResponse:
        return SyncAudiencesResponse()
