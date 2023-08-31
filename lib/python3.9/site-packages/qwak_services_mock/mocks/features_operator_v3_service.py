import uuid
from typing import Optional

from _qwak_proto.qwak.features_operator.v3.features_operator_async_service_pb2 import (
    ValidationResponse,
)
from _qwak_proto.qwak.features_operator.v3.features_operator_async_service_pb2_grpc import (
    FeaturesOperatorAsyncServiceServicer,
)


class FeaturesOperatorV3ServiceMock(FeaturesOperatorAsyncServiceServicer):
    def __init__(self):
        self.response: Optional[ValidationResponse] = None
        super(FeaturesOperatorV3ServiceMock, self).__init__()

    def given_next_response(self, response: ValidationResponse):
        self.response = response

    def ValidateDataSource(self, request, context) -> str:
        request_id = str(uuid.uuid4())
        return ValidationResponse(request_id=request_id)

    def ValidateFeatureSet(self, request, context):
        request_id = str(uuid.uuid4())
        return ValidationResponse(request_id=request_id)

    def GetValidationResult(self, request, context):
        return self.response
