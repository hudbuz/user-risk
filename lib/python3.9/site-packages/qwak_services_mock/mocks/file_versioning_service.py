from collections import defaultdict
from typing import Dict, List

from _qwak_proto.qwak.file_versioning.file_versioning_pb2 import FileTagSpec
from _qwak_proto.qwak.file_versioning.file_versioning_service_pb2 import (
    GetModelFileTagsRequest,
    GetModelFileTagsResponse,
    RegisterFileTagRequest,
    RegisterFileTagResponse,
)
from _qwak_proto.qwak.file_versioning.file_versioning_service_pb2_grpc import (
    FileVersioningManagementServiceServicer,
)
from qwak_services_mock.mocks.utils.exception_handlers import raise_internal_grpc_error


class FileVersioningServiceMock(FileVersioningManagementServiceServicer):
    def __init__(self):
        super(FileVersioningServiceMock, self).__init__()
        self.tags: Dict[str : List[FileTagSpec]] = defaultdict(list)

    def RegisterFileTag(
        self, request: RegisterFileTagRequest, context
    ) -> RegisterFileTagResponse:
        try:
            self.tags[request.file_tag_spec.model_id].append(request.file_tag_spec)
            return RegisterFileTagResponse()
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def GetModelFileTags(
        self, request: GetModelFileTagsRequest, context
    ) -> GetModelFileTagsResponse:
        try:
            return GetModelFileTagsResponse(file_tags=self.tags[request.model_id])
        except Exception as e:
            raise_internal_grpc_error(context, e)
