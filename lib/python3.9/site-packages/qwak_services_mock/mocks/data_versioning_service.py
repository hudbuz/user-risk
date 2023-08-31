from collections import defaultdict
from typing import Dict, List

from _qwak_proto.qwak.data_versioning.data_versioning_pb2 import DataTagSpec
from _qwak_proto.qwak.data_versioning.data_versioning_service_pb2 import (
    GetModelDataTagsRequest,
    GetModelDataTagsResponse,
    RegisterDataTagRequest,
    RegisterDataTagResponse,
)
from _qwak_proto.qwak.data_versioning.data_versioning_service_pb2_grpc import (
    DataVersioningManagementServiceServicer,
)
from qwak_services_mock.mocks.utils.exception_handlers import raise_internal_grpc_error


class DataVersioningServiceMock(DataVersioningManagementServiceServicer):
    def __init__(self):
        super(DataVersioningServiceMock, self).__init__()
        self.tags: Dict[str : List[DataTagSpec]] = defaultdict(list)

    def RegisterDataTag(
        self, request: RegisterDataTagRequest, context
    ) -> RegisterDataTagResponse:
        try:
            self.tags[request.data_tag_spec.model_id].append(request.data_tag_spec)
            return RegisterDataTagResponse()
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def GetModelDataTags(
        self, request: GetModelDataTagsRequest, context
    ) -> GetModelDataTagsResponse:
        try:
            return GetModelDataTagsResponse(data_tags=self.tags[request.model_id])
        except Exception as e:
            raise_internal_grpc_error(context, e)
