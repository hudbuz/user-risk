from typing import List

from _qwak_proto.qwak.audience.v1.audience_api_pb2 import (
    CreateAudienceRequest,
    CreateAudienceResponse,
    DeleteAudienceRequest,
    GetAudienceRequest,
    GetAudienceResponse,
    ListAudienceRequest,
    ListAudienceResponse,
    UpdateAudienceRequest,
)
from _qwak_proto.qwak.audience.v1.audience_api_pb2_grpc import AudienceAPIStub
from _qwak_proto.qwak.audience.v1.audience_pb2 import Audience, AudienceEntry
from dependency_injector.wiring import Provide
from qwak.inner.di_configuration import QwakContainer


class AudienceClient:
    def __init__(self, grpc_channel=Provide[QwakContainer.core_grpc_channel]):
        self._service_stub = AudienceAPIStub(grpc_channel)

    def create_audience(self, audience: Audience) -> str:
        create_audience_request = CreateAudienceRequest(audience=audience)
        create_audience_response: CreateAudienceResponse = (
            self._service_stub.CreateAudience(create_audience_request)
        )
        return create_audience_response.audience_id

    def get_audience(self, audience_id: str) -> Audience:
        get_audience_request = GetAudienceRequest(audience_id=audience_id)
        get_audience_response: GetAudienceResponse = self._service_stub.GetAudience(
            get_audience_request
        )
        return get_audience_response.audience

    def delete_audience(self, audience_id: str) -> None:
        delete_audience_request = DeleteAudienceRequest(audience_id=audience_id)
        self._service_stub.DeleteAudience(delete_audience_request)

    def list_audience(self) -> List[AudienceEntry]:
        list_audience_request = ListAudienceRequest()
        list_audience_response: ListAudienceResponse = self._service_stub.ListAudience(
            list_audience_request
        )
        return list_audience_response.audience_entries

    def update_audience(self, audience_id: str, audience: Audience) -> None:
        update_audience_request = UpdateAudienceRequest(
            id=audience_id, audience=audience
        )
        self._service_stub.UpdateAudience(update_audience_request)
