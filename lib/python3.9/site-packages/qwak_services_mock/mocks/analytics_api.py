from _qwak_proto.qwak.analytics.analytics_pb2 import QueryStatus
from _qwak_proto.qwak.analytics.analytics_service_pb2 import (
    GetQueryResultDownloadURLResponse,
    GetQueryResultsPageResponse,
    QueryResponse,
)
from _qwak_proto.qwak.analytics.analytics_service_pb2_grpc import (
    AnalyticsQueryServiceServicer,
)

SUCCESSFUL_QUERY = "select * from successful"
FAILED_QUERY = "select * from failed"
TIMEOUT_QUERY = "select * from timeout"
CANCELLED_QUERY = "select * from cancelled"
SUCCESSFUL_QUERY_ID = "successful"
FAILED_QUERY_ID = "failed"
TIMEOUT_QUERY_ID = "timeout"
CANCELLED_QUERY_ID = "cancelled"
FAILURE_REASON = "failure reason"
DOWNLOAD_URL = "https://qwak.com/download"


class AnalyticsApiMock(AnalyticsQueryServiceServicer):
    def __init__(self):
        super().__init__()

    def Query(self, request, context):
        query = request.query
        if query == SUCCESSFUL_QUERY:
            return QueryResponse(query_id=SUCCESSFUL_QUERY_ID)
        elif query == FAILED_QUERY:
            return QueryResponse(query_id=FAILED_QUERY_ID)
        elif query == TIMEOUT_QUERY:
            return QueryResponse(query_id=TIMEOUT_QUERY_ID)
        elif query == CANCELLED_QUERY:
            return QueryResponse(query_id=CANCELLED_QUERY_ID)
        else:
            return QueryResponse(query_id="unknown")

    def GetQueryResultsPage(self, request, context):
        query_id = request.query_id
        if query_id == SUCCESSFUL_QUERY_ID:
            return GetQueryResultsPageResponse(status=QueryStatus.SUCCESS, data=None)
        elif query_id == FAILED_QUERY_ID:
            return GetQueryResultsPageResponse(
                status=QueryStatus.FAILED, failure_reason=FAILURE_REASON
            )
        elif query_id == TIMEOUT_QUERY_ID:
            return GetQueryResultsPageResponse(status=QueryStatus.PENDING, data=None)
        elif query_id == CANCELLED_QUERY_ID:
            return GetQueryResultsPageResponse(status=QueryStatus.CANCELED, data=None)

    def GetQueryResultDownloadURL(self, request, context):
        return GetQueryResultDownloadURLResponse(download_url=DOWNLOAD_URL)
