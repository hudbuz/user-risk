from collections import defaultdict
from enum import Enum
from typing import Dict, List

from _qwak_proto.qwak.logging.log_line_pb2 import LogLine
from _qwak_proto.qwak.logging.log_reader_service_pb2 import (
    ReadLogsRequest,
    ReadLogsResponse,
)
from _qwak_proto.qwak.logging.log_reader_service_pb2_grpc import (
    LogReaderServiceServicer,
)
from _qwak_proto.qwak.logging.log_source_pb2 import (
    InferenceExecutionSource,
    LogSource,
    ModelRuntimeSource,
    RemoteBuildSource,
    StreamingAggregationFeatureSetSource,
    StreamingFeatureSetSource,
)


class Source(Enum):
    MODEL_RUNTIME = 1
    REMOTE_BUILD = 2
    INFERENCE_EXECUTION = 3
    STREAMING_AGGREGATION = 4
    STREAMING_FEATURE_SET = 5


class MockLogLine:
    def __init__(self, log_line: LogLine, metadata: Dict[str, any]):
        self.log_line = log_line
        self.metadata = metadata


class LoggingServiceApiMock(LogReaderServiceServicer):
    def __init__(self):
        self._logs: Dict[Source, List[MockLogLine]] = defaultdict(list)
        super(LoggingServiceApiMock, self).__init__()

    def ReadLogs(self, request: ReadLogsRequest, context) -> ReadLogsResponse:
        source_to_use = self.__get_source_from_request(request.source)
        logs = []
        if source_to_use == Source.MODEL_RUNTIME:
            logs = self.__read_model_runtime_logs(request)
        elif source_to_use == Source.INFERENCE_EXECUTION:
            logs = self.__read_inference_execution_logs(request)
        elif source_to_use == Source.REMOTE_BUILD:
            logs = self.__read_remote_build_logs(request)
        elif source_to_use == Source.STREAMING_AGGREGATION:
            logs = self.__read_streaming_aggregation_logs(request)
        elif source_to_use == Source.STREAMING_FEATURE_SET:
            logs = self.__read_streaming_feature_set_logs(request)

        if request.after_offset:
            logs = list(
                filter(
                    lambda mocked_log: mocked_log.metadata.get("offset")
                    >= request.after_offset,
                    logs,
                )
            )
        if request.before_offset:
            logs = list(
                filter(
                    lambda mocked_log: mocked_log.metadata.get("offset")
                    < request.before_offset,
                    logs,
                )
            )
        logs.sort(
            key=lambda mocked_log: (
                mocked_log.metadata.get("offset") is None,
                mocked_log.metadata.get("offset"),
            ),
            reverse=True,
        )

        has_next_page = False
        if request.max_number_of_results > 0:
            has_next_page = len(logs) > request.max_number_of_results
            logs = logs[: request.max_number_of_results]
        try:
            first_offset = min(
                list(map(lambda mocked_log: mocked_log.metadata.get("offset"), logs))
            )
            last_offset = max(
                list(map(lambda mocked_log: mocked_log.metadata.get("offset"), logs))
            )
        except Exception:
            first_offset = "0"
            last_offset = "0"

        return ReadLogsResponse(
            log_line=list(map(lambda mocked_log: mocked_log.log_line, logs)),
            first_offset=first_offset,
            last_offset=last_offset,
            has_next_page=has_next_page,
        )

    @staticmethod
    def _get_source(source: LogSource):
        if type(source) is ModelRuntimeSource:
            return Source.MODEL_RUNTIME
        elif type(source) is RemoteBuildSource:
            return Source.REMOTE_BUILD
        elif type(source) is InferenceExecutionSource:
            return Source.INFERENCE_EXECUTION
        elif type(source) is StreamingFeatureSetSource:
            return Source.STREAMING_AGGREGATION
        elif type(source) is StreamingAggregationFeatureSetSource:
            return Source.STREAMING_FEATURE_SET
        else:
            raise Exception("Unknown source")

    @staticmethod
    def __get_source_from_request(source: LogSource):
        request_source = source.WhichOneof("source")
        if request_source == "model_runtime":
            return Source.MODEL_RUNTIME
        elif request_source == "remote_build":
            return Source.REMOTE_BUILD
        elif request_source == "inference_execution":
            return Source.INFERENCE_EXECUTION
        elif request_source == "streaming_aggregation_feature_set":
            return Source.STREAMING_AGGREGATION
        elif request_source == "streaming_feature_set":
            return Source.STREAMING_FEATURE_SET
        else:
            raise Exception("Unknown source")

    def __read_model_runtime_logs(self, request: ReadLogsRequest) -> List[MockLogLine]:
        logs: List[MockLogLine] = self._logs[Source.MODEL_RUNTIME]
        if request.source.model_runtime.deployment_id:
            logs = list(
                filter(
                    lambda mocked_log: mocked_log.metadata.get("deployment_id")
                    == request.source.model_runtime.deployment_id,
                    logs,
                )
            )
        elif request.source.model_runtime.build_id:
            logs = list(
                filter(
                    lambda mocked_log: mocked_log.metadata.get("build_id")
                    == request.source.model_runtime.build_id,
                    logs,
                )
            )

        return logs

    def __read_inference_execution_logs(self, request) -> List[MockLogLine]:
        logs: List[MockLogLine] = self._logs[Source.INFERENCE_EXECUTION]
        if request.source.inference_execution.inference_job_id:
            logs = list(
                filter(
                    lambda mocked_log: mocked_log.metadata.get("inference_job_id")
                    == request.source.inference_execution.inference_job_id,
                    logs,
                )
            )
        elif request.source.inference_execution.inference_task_id:
            logs = list(
                filter(
                    lambda mocked_log: mocked_log.metadata.get("inference_task_id")
                    == request.source.inference_execution.inference_task_id,
                    logs,
                )
            )

        return logs

    def __read_remote_build_logs(self, request) -> List[MockLogLine]:
        logs: List[MockLogLine] = self._logs[Source.REMOTE_BUILD]
        if request.source.remote_build.build_id:
            logs = list(
                filter(
                    lambda mocked_log: mocked_log.metadata.get("build_id")
                    == request.source.remote_build.build_id,
                    logs,
                )
            )
        if request.source.remote_build.phase_ids:
            logs = list(
                filter(
                    lambda mocked_log: mocked_log.metadata.get("phase_id")
                    in request.source.remote_build.phase_ids,
                    logs,
                )
            )

        return logs

    def __read_streaming_aggregation_logs(self, request):
        raise Exception("Unimplemented")

    def __read_streaming_feature_set_logs(self, request):
        raise Exception("Unimplemented")

    def add_mock_logs(self, source: LogSource, rows: List[MockLogLine]):
        self._logs[self._get_source(source)].extend(rows)
