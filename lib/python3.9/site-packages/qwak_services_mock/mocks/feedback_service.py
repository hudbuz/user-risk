from _qwak_proto.qwak.inference.feedback.feedback_pb2 import (
    ActualValuesRequest,
    ActualValuesResponse,
    ConfigureFeedbackRequest,
    ConfigureFeedbackResponse,
)
from _qwak_proto.qwak.inference.feedback.feedback_pb2_grpc import (
    FeedbackServiceServicer,
)
from qwak_services_mock.mocks.utils.exception_handlers import raise_internal_grpc_error


class FeedbackServiceMock(FeedbackServiceServicer):
    def __init__(self):
        super(FeedbackServiceMock, self).__init__()

    def PostFeedback(
        self, request: ActualValuesRequest, context
    ) -> ActualValuesResponse:
        try:
            return ActualValuesResponse(status=ActualValuesResponse.ValueStatus.SUCCESS)
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def ConfigureFeedback(
        self, request: ConfigureFeedbackRequest, context
    ) -> ConfigureFeedbackResponse:
        try:
            return ConfigureFeedbackResponse(
                status=ConfigureFeedbackResponse.ValueStatus.SUCCESS
            )
        except Exception as e:
            raise_internal_grpc_error(context, e)
