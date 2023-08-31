import uuid
from typing import Dict, List, Optional, Set

from _qwak_proto.qwak.administration.authenticated_user.v1.credentials_pb2 import (
    AwsTemporaryCredentials,
)
from _qwak_proto.qwak.batch_job.v1.batch_job_service_pb2 import (
    BatchJobDeploymentSize,
    BatchJobDetails,
    BatchJobMessage,
    BatchJobStatusMessage,
    CancelBatchJobRequest,
    CancelBatchJobResponse,
    CancelWarmupJobRequest,
    CancelWarmupJobResponse,
    ExecutionReportLine,
    GetBatchJobDetailsRequest,
    GetBatchJobDetailsResponse,
    GetBatchJobDownloadDetailsRequest,
    GetBatchJobDownloadDetailsResponse,
    GetBatchJobPreSignedDownloadUrlRequest,
    GetBatchJobPreSignedDownloadUrlResponse,
    GetBatchJobPreSignedUploadUrlRequest,
    GetBatchJobPreSignedUploadUrlResponse,
    GetBatchJobReportRequest,
    GetBatchJobReportResponse,
    GetBatchJobStatusRequest,
    GetBatchJobStatusResponse,
    GetBatchJobUploadDetailsRequest,
    GetBatchJobUploadDetailsResponse,
    ListBatchJobsRequest,
    ListBatchJobsResponse,
    StartBatchJobRequest,
    StartBatchJobResponse,
    StartWarmupJobRequest,
    StartWarmupJobResponse,
)
from _qwak_proto.qwak.batch_job.v1.batch_job_service_pb2_grpc import (
    BatchJobManagementServiceServicer,
)
from google.protobuf.timestamp_pb2 import Timestamp
from qwak_services_mock.mocks.utils.exception_handlers import raise_internal_grpc_error

BATCH_JOB_NOT_FOUND_ERROR = "Batch Job Not Found"


class BatchJobManagerService(BatchJobManagementServiceServicer):
    def __init__(self):
        self.id_to_batch_job: Dict[str, MockBatchJob] = dict()
        self.model_id_active_warmup: Set[str] = set()
        self.models_to_fail: Set[str] = set()

    def StartBatchJob(
        self, request: StartBatchJobRequest, context
    ) -> StartBatchJobResponse:
        try:
            model_id = (
                request.model_id
                if request.model_id
                else request.batch_job_request.model_details.model_id
            )

            if model_id in self.models_to_fail:
                return StartBatchJobResponse(
                    success=False, failure_message="Model start failed"
                )
            mock_batch_job: MockBatchJob = MockBatchJob(request)
            self.id_to_batch_job[mock_batch_job.id] = mock_batch_job
            return StartBatchJobResponse(batch_id=mock_batch_job.id, success=True)
        except Exception as e:
            return StartBatchJobResponse(success=False, failure_message=str(e))

    def CancelBatchJob(
        self, request: CancelBatchJobRequest, context
    ) -> CancelBatchJobResponse:
        try:
            if request.batch_id in self.id_to_batch_job.keys():
                self.id_to_batch_job[
                    request.batch_id
                ].status = BatchJobStatusMessage.BATCH_JOB_CANCELLED_STATUS
                return CancelBatchJobResponse(success=True)
            else:
                return CancelBatchJobResponse(
                    success=False, failure_message=BATCH_JOB_NOT_FOUND_ERROR
                )
        except Exception as e:
            return CancelBatchJobResponse(success=False, failure_message=str(e))

    def StartWarmupJob(
        self, request: StartWarmupJobRequest, context
    ) -> StartWarmupJobResponse:
        try:
            self.model_id_active_warmup.add(request.model_id)
            return StartWarmupJobResponse(success=True)
        except Exception as e:
            return StartWarmupJobResponse(success=False, failure_message=str(e))

    def CancelWarmupJob(
        self, request: CancelWarmupJobRequest, context
    ) -> CancelWarmupJobResponse:
        try:
            if request.model_id in self.model_id_active_warmup:
                self.model_id_active_warmup.remove(request.model_id)
            return CancelWarmupJobResponse(success=True)
        except Exception as e:
            return CancelWarmupJobResponse(success=False, failure_message=str(e))

    def GetBatchJobStatus(
        self, request: GetBatchJobStatusRequest, context
    ) -> GetBatchJobStatusResponse:
        try:
            if request.batch_id in self.id_to_batch_job:
                batch_job = self.id_to_batch_job[request.batch_id]
                return GetBatchJobStatusResponse(
                    success=True,
                    job_status=batch_job.status,
                    finished_files=batch_job.finished_files,
                    total_files=batch_job.total_files,
                )
            else:
                return GetBatchJobStatusResponse(
                    success=False, failure_message=BATCH_JOB_NOT_FOUND_ERROR
                )
        except Exception as e:
            return GetBatchJobStatusResponse(success=False, failure_message=str(e))

    def GetBatchJobReport(
        self, request: GetBatchJobReportRequest, context
    ) -> GetBatchJobReportResponse:
        try:
            if request.batch_id in self.id_to_batch_job:
                batch_job = self.id_to_batch_job[request.batch_id]
                return GetBatchJobReportResponse(
                    successful=True, report_messages=batch_job.report
                )
            else:
                return GetBatchJobReportResponse(
                    successful=False, failure_message=BATCH_JOB_NOT_FOUND_ERROR
                )
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def ListBatchJobs(
        self, request: ListBatchJobsRequest, context
    ) -> ListBatchJobsResponse:
        try:
            batch_jobs = list(
                filter(
                    lambda job: job.start_request.model_id == request.model_id,
                    self.id_to_batch_job.values(),
                )
            )
            batch_jobs_details = list(map(batch_job_to_details, batch_jobs))
            return ListBatchJobsResponse(batch_jobs=batch_jobs_details)
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def GetBatchJobDetails(
        self, request: GetBatchJobDetailsRequest, context
    ) -> GetBatchJobDetailsResponse:
        try:
            if request.job_id in self.id_to_batch_job:
                return GetBatchJobDetailsResponse(
                    success=True,
                    batch_job=batch_job_to_message(
                        self.id_to_batch_job[request.job_id]
                    ),
                )
            else:
                return GetBatchJobDetailsResponse(
                    success=False, failure_message=BATCH_JOB_NOT_FOUND_ERROR
                )
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def GetBatchJobPreSignedUploadUrl(
        self, request: GetBatchJobPreSignedUploadUrlRequest, context
    ) -> GetBatchJobPreSignedUploadUrlResponse:
        try:
            return GetBatchJobPreSignedUploadUrlResponse(
                input_path="/input_path",
                output_path="/output_path",
                bucket="bucket",
                urls=[
                    f"{request.model_id}_{i}.{request.file_type}"
                    for i in range(request.number_of_files)
                ],
                success=True,
            )
        except Exception as e:
            return GetBatchJobPreSignedUploadUrlResponse(
                success=False, failure_message=str(e)
            )

    def GetBatchJobPreSignedDownloadUrl(
        self, request: GetBatchJobPreSignedDownloadUrlRequest, context
    ) -> GetBatchJobPreSignedDownloadUrlResponse:
        try:
            if request.job_id in self.id_to_batch_job:
                batch_job = self.id_to_batch_job[request.job_id]
                return GetBatchJobPreSignedDownloadUrlResponse(
                    urls=[
                        f"bucket://output_path/{batch_job.id}/{i}.{batch_job.start_request.output_file_type}"
                        for i in range(batch_job.total_files)
                    ],
                    success=True,
                )
            else:
                return GetBatchJobPreSignedDownloadUrlResponse(
                    success=False, failure_message=BATCH_JOB_NOT_FOUND_ERROR
                )
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def GetBatchJobUploadDetails(
        self, request: GetBatchJobUploadDetailsRequest, context
    ) -> GetBatchJobUploadDetailsResponse:
        try:
            return GetBatchJobUploadDetailsResponse(
                input_path="/input_path",
                output_path="/output_path",
                bucket="bucket",
                credentials=AwsTemporaryCredentials(
                    access_key_id="access_key_id",
                    secret_access_key="secret_access_key",
                    session_token="session_token",
                    expiration_time=Timestamp(seconds=3600),
                ),
            )
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def GetBatchJobDownloadDetails(
        self, request: GetBatchJobDownloadDetailsRequest, context
    ) -> GetBatchJobDownloadDetailsResponse:
        try:
            if request.job_id in self.id_to_batch_job:
                batch_job = self.id_to_batch_job[request.job_id]
                return GetBatchJobDownloadDetailsResponse(
                    bucket="bucket",
                    keys=[
                        f"bucket://output_path/{batch_job.id}/{i}.{batch_job.start_request.output_file_type}"
                        for i in range(batch_job.total_files)
                    ],
                    credentials=AwsTemporaryCredentials(
                        access_key_id="access_key_id",
                        secret_access_key="secret_access_key",
                        session_token="session_token",
                        expiration_time=Timestamp(seconds=3600),
                    ),
                )
            else:
                raise Exception(BATCH_JOB_NOT_FOUND_ERROR)
        except Exception as e:
            raise_internal_grpc_error(context, e)


class MockBatchJob:
    def __init__(self, request: StartBatchJobRequest):
        self.start_request: StartBatchJobRequest = request
        self.status: BatchJobStatusMessage = (
            BatchJobStatusMessage.BATCH_JOB_PENDING_STATUS
        )
        self.start_time: Timestamp = Timestamp()
        self.end_time: Optional[Timestamp] = None
        self.id: str = str(uuid.uuid4())
        self.finished_files: int = 0
        self.total_files: int = 0
        self.report: List[str] = []

    def add_report_messages(self, messages: List[str]):
        self.report.extend(messages)


def batch_job_to_details(batch_job: MockBatchJob) -> BatchJobDetails:
    return BatchJobDetails(
        build_id=batch_job.start_request.build_id,
        job_id=batch_job.id,
        job_status=batch_job.status,
        start_time=batch_job.start_time,
        end_time=batch_job.end_time,
        job_size=BatchJobDeploymentSize(
            number_of_pods=batch_job.start_request.batch_job_deployment_size.number_of_pods,
            cpu=batch_job.start_request.batch_job_deployment_size.cpu,
            memory_amount=batch_job.start_request.batch_job_deployment_size.memory_amount,
            memory_units=batch_job.start_request.batch_job_deployment_size.memory_units,
        ),
    )


def batch_job_to_message(batch_job: MockBatchJob) -> BatchJobMessage:
    return BatchJobMessage(
        job_id=batch_job.id,
        model_id=batch_job.start_request.model_id,
        branch_id=batch_job.start_request.branch_id,
        build_id=batch_job.start_request.build_id,
        start_time=batch_job.start_time,
        job_status=batch_job.status,
        end_time=batch_job.end_time,
        job_size=BatchJobDeploymentSize(
            number_of_pods=batch_job.start_request.batch_job_deployment_size.number_of_pods,
            cpu=batch_job.start_request.batch_job_deployment_size.cpu,
            memory_amount=batch_job.start_request.batch_job_deployment_size.memory_amount,
            memory_units=batch_job.start_request.batch_job_deployment_size.memory_units,
        ),
        report_messages=[
            ExecutionReportLine(time=Timestamp(), line=line)
            for line in batch_job.report
        ],
    )
