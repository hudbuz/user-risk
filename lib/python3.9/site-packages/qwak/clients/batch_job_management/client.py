import grpc
from _qwak_proto.qwak.batch_job.v1.batch_job_service_pb2 import (
    AdvancedDeploymentOptions,
    BatchJobDataDetails,
    BatchJobDeploymentSize,
    BatchJobDestinationPath,
    BatchJobExecutionDetails,
    BatchJobModelDetails,
    BatchJobParameter,
    BatchJobRequest,
    BatchJobSourcePath,
    BatchJobStatusMessage,
    CancelBatchJobRequest,
    CancelBatchJobResponse,
    CancelWarmupJobRequest,
    CancelWarmupJobResponse,
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
    StartBatchJobRequest,
    StartBatchJobResponse,
    StartWarmupJobRequest,
    StartWarmupJobResponse,
)
from _qwak_proto.qwak.batch_job.v1.batch_job_service_pb2_grpc import (
    BatchJobManagementServiceStub,
)
from _qwak_proto.qwak.deployment.deployment_pb2 import MemoryUnit
from _qwak_proto.qwak.logging.log_reader_service_pb2 import ReadLogsResponse
from _qwak_proto.qwak.user_application.common.v0.resources_pb2 import (
    ClientPodComputeResources,
    CpuResources,
    GpuResources,
    PodComputeResourceTemplateSpec,
)
from dependency_injector.wiring import Provide
from qwak.clients.logging_client import LoggingClient
from qwak.clients.model_management import ModelsManagementClient
from qwak.exceptions import QwakException
from qwak.inner.di_configuration import QwakContainer

from .executions_config import (
    INPUT_FORMATTERS_MAP,
    OUTPUT_FORMATTERS_MAP,
    ExecutionConfig,
)
from .results import (
    CancelExecutionResult,
    ExecutionStatusResult,
    GetBatchJobPreSignedDownloadUrlResult,
    GetBatchJobPreSignedUploadUrlResult,
    GetExecutionReportResult,
    StartExecutionResult,
)

CLIENT_TIMEOUT = 180  # Seconds


class BatchJobManagerClient:
    def __init__(
        self,
        grpc_channel=Provide[QwakContainer.core_grpc_channel],
        logging_client: LoggingClient = None,
    ):
        self.batch_job_management = BatchJobManagementServiceStub(grpc_channel)
        self.logging_client = logging_client

    def start_execution(
        self,
        execution_config: ExecutionConfig,
    ) -> StartExecutionResult:
        """

        Args:
            execution_config: The configuration that hold details for the batch execution (cloud locations, concurrency,
            etc...

        Returns:
            The response received from the api. On successful start of execution a batch job id is returned
        """
        job_size = self.get_batch_deployment_size_from_resources(execution_config)

        try:
            user_raw_input_format = execution_config.execution.input_file_type.upper()
            user_raw_output_format = execution_config.execution.output_file_type.upper()

            if user_raw_input_format not in INPUT_FORMATTERS_MAP:
                raise ValueError(
                    f"Invalid input format - please choose one of {list(INPUT_FORMATTERS_MAP.keys())}"
                )

            if user_raw_output_format not in OUTPUT_FORMATTERS_MAP:
                raise ValueError(
                    f"Invalid output format - please choose one of {list(OUTPUT_FORMATTERS_MAP.keys())}"
                )

            start_job_result: StartBatchJobResponse = self._start_batch_job(
                batch_job_request=BatchJobRequest(
                    model_details=BatchJobModelDetails(
                        model_id=execution_config.execution.model_id,
                        build_id=execution_config.execution.build_id,
                    ),
                    data_details=BatchJobDataDetails(
                        source_path=BatchJobSourcePath(
                            source_folder=execution_config.execution.source_folder,
                            source_bucket=execution_config.execution.source_bucket,
                            input_file_type=INPUT_FORMATTERS_MAP.get(
                                user_raw_input_format
                            ),
                        ),
                        destination_path=BatchJobDestinationPath(
                            destination_bucket=execution_config.execution.destination_bucket,
                            destination_folder=execution_config.execution.destination_folder,
                            output_file_type=OUTPUT_FORMATTERS_MAP.get(
                                user_raw_output_format
                            ),
                        ),
                        token_secret=execution_config.execution.access_token_name,
                        secret_secret=execution_config.execution.access_secret_name,
                    ),
                    execution_details=BatchJobExecutionDetails(
                        job_timeout=execution_config.execution.job_timeout,
                        task_timeout=execution_config.execution.file_timeout,
                        batch_job_deployment_size=job_size,
                        advanced_deployment_options=AdvancedDeploymentOptions(
                            custom_iam_role_arn=execution_config.advanced_options.custom_iam_role_arn
                        ),
                        parameters=BatchJobManagerClient._batch_job_parameters_as_list(
                            execution_config.execution.parameters
                        ),
                    ),
                )
            )

            return StartExecutionResult(
                success=start_job_result.success,
                execution_id=start_job_result.batch_id,
                failure_message=start_job_result.failure_message,
            )
        except grpc.RpcError as e:
            raise QwakException(f"Failed to start execution, error is {e}")

    @staticmethod
    def get_batch_deployment_size_from_resources(execution_config):
        if execution_config.resources.instance_size:
            return BatchJobDeploymentSize(
                number_of_pods=execution_config.resources.pods,
                client_pod_compute_resources=ClientPodComputeResources(
                    template_spec=PodComputeResourceTemplateSpec(
                        template_id=execution_config.resources.instance_size,
                    )
                ),
            )

        if execution_config.resources.gpu_amount > 0:
            return BatchJobDeploymentSize(
                number_of_pods=execution_config.resources.pods,
                client_pod_compute_resources=ClientPodComputeResources(
                    gpu_resources=GpuResources(
                        gpu_type=execution_config.resources.gpu_type,
                        gpu_amount=execution_config.resources.gpu_amount,
                    )
                ),
            )

        return BatchJobDeploymentSize(
            number_of_pods=execution_config.resources.pods,
            client_pod_compute_resources=ClientPodComputeResources(
                cpu_resources=CpuResources(
                    cpu=execution_config.resources.cpus,
                    memory_amount=execution_config.resources.memory,
                    memory_units=MemoryUnit.MIB,
                )
            ),
        )

    def _start_batch_job(
        self,
        batch_job_request: BatchJobRequest,
    ) -> StartBatchJobResponse:
        """

        Args:
            batch_job_request: The Api for batch job request

        Returns:
            The response received from the api. On successful start of execution a batch job id is returned
        """
        return self.batch_job_management.StartBatchJob(
            StartBatchJobRequest(batch_job_request=batch_job_request),
            timeout=CLIENT_TIMEOUT,
        )

    @staticmethod
    def _batch_job_parameters_as_list(params: dict):
        return [
            BatchJobParameter(key=key, value=value) for (key, value) in params.items()
        ]

    def start_warmup_job(
        self, execution_config: ExecutionConfig
    ) -> StartWarmupJobResponse:
        """

        Args:
            execution_config: The configuration that hold details for the batch execution (resources, concurrency,
            etc...

        Returns:
            The response received from the api. On successful start of execution a batch job id is returned
        """
        job_size = self.get_batch_deployment_size_from_resources(execution_config)

        try:
            start_warmup_job_result: StartWarmupJobRequest = (
                self.batch_job_management.StartWarmupJob(
                    StartWarmupJobRequest(
                        model_id=execution_config.execution.model_id,
                        build_id=execution_config.execution.build_id,
                        branch_id=ModelsManagementClient().get_model_uuid(
                            execution_config.execution.model_id,
                        ),
                        warmup_timeout=execution_config.warmup.timeout,
                        batch_job_deployment_size=job_size,
                    ),
                    timeout=CLIENT_TIMEOUT,
                )
            )

            return StartWarmupJobResponse(
                success=start_warmup_job_result.success,
                failure_message=start_warmup_job_result.failure_message,
            )
        except grpc.RpcError as e:
            raise QwakException(f"Failed to start warmup, error is {e}")

    def get_execution_status(self, execution_id: str) -> ExecutionStatusResult:
        """

        Args:
            execution_id: the batch execution id to get the current status of

        Returns:
            the status of the execution

        """
        try:
            batch_job_status: GetBatchJobStatusResponse = (
                self.batch_job_management.GetBatchJobStatus(
                    GetBatchJobStatusRequest(
                        batch_id=execution_id,
                    )
                )
            )
            return ExecutionStatusResult(
                success=batch_job_status.success,
                status=BatchJobStatusMessage.Name(batch_job_status.job_status),
                finished_files=batch_job_status.finished_files,
                total_files=batch_job_status.total_files,
                failure_message=batch_job_status.failure_message,
            )
        except grpc.RpcError as e:
            raise QwakException(
                f"Failed to get execution status, error is {e.details()}"
            )

    def cancel_warmup(
        self, execution_config: ExecutionConfig
    ) -> CancelWarmupJobResponse:
        """

        Args:
            execution_config: The configuration that hold details for the batch execution (resources, concurrency,
            etc...

        Returns:
            The response received from the api. On successful start of execution a batch job id is returned
        """
        try:
            cancel_warmup_job_response: CancelWarmupJobResponse = (
                self.batch_job_management.CancelWarmupJob(
                    CancelWarmupJobRequest(
                        model_id=execution_config.execution.model_id,
                        build_id=execution_config.execution.build_id,
                        branch_id=ModelsManagementClient().get_model_uuid(
                            execution_config.execution.model_id,
                        ),
                    ),
                    timeout=CLIENT_TIMEOUT,
                )
            )

            return CancelWarmupJobResponse(
                success=cancel_warmup_job_response.success,
                failure_message=cancel_warmup_job_response.failure_message,
            )

        except grpc.RpcError as e:
            raise QwakException(f"Failed to cancel execution, error is {e}")

    def cancel_execution(self, execution_id: str) -> CancelExecutionResult:
        """

        Args:
            execution_id: the batch execution id to get the current status of

        Returns:
            A successful response or failure of the cancel process

        """
        try:
            cancel_batch_job_response: CancelBatchJobResponse = (
                self.batch_job_management.CancelBatchJob(
                    CancelBatchJobRequest(
                        batch_id=execution_id,
                    ),
                    timeout=CLIENT_TIMEOUT,
                )
            )

            return CancelExecutionResult(
                success=cancel_batch_job_response.success,
                failure_message=cancel_batch_job_response.failure_message,
            )

        except grpc.RpcError as e:
            raise QwakException(f"Failed to cancel execution, error is {e}")

    def get_execution_report(self, execution_id: str) -> GetExecutionReportResult:
        """

        Args:
            execution_id: the batch execution id to get the current status of

        Returns:
            A full report of all the events that occurred as part of the execution job

        """
        try:
            self.logging_client = (
                self.logging_client if self.logging_client else LoggingClient()
            )
            batch_job_report: GetBatchJobReportResponse = (
                self.batch_job_management.GetBatchJobReport(
                    GetBatchJobReportRequest(batch_id=execution_id),
                    timeout=CLIENT_TIMEOUT,
                )
            )

            try:
                execution_log_response: ReadLogsResponse = (
                    self.logging_client.read_execution_models_logs(
                        execution_id=execution_id
                    )
                )
                execution_logs = execution_log_response.log_line
                execution_logs.sort(key=lambda line: line.ingested_iso_timestamp)
                execution_logs_msgs = [line.text for line in execution_logs]
            except QwakException as e:
                execution_logs_msgs = [
                    f"Error reading the execution model run logs due to: {e.message}"
                ]

            return GetExecutionReportResult(
                success=batch_job_report.successful,
                failure_message=batch_job_report.failure_message,
                records=batch_job_report.report_messages,
                model_logs=execution_logs_msgs,
            )

        except grpc.RpcError as e:
            raise QwakException(f"Failed to get report for execution, error is: {e}")

    def get_pre_signed_upload_urls_details(
        self, model_id: str, number_of_batches: int, file_type: str = "csv"
    ) -> GetBatchJobPreSignedUploadUrlResult:
        """
        Get pre signed upload urls details in order to start a job using Qwak cloud bucket
        Args:
            model_id: The model id for the execution
            number_of_batches: The number of pre-signed urls to request
            file_type: The file type that will be uploaded

        Returns: GetBatchJobPreSignedUploadUrlResult which contains the input/output path, the bucket, and the pre-signed urls
        """
        try:
            response: GetBatchJobPreSignedUploadUrlResponse = (
                self.batch_job_management.GetBatchJobPreSignedUploadUrl(
                    GetBatchJobPreSignedUploadUrlRequest(
                        model_id=model_id,
                        number_of_files=number_of_batches,
                        file_type=file_type,
                    ),
                    timeout=CLIENT_TIMEOUT,
                )
            )

            return GetBatchJobPreSignedUploadUrlResult(
                success=response.success,
                failure_message=response.failure_message,
                input_path=response.input_path,
                output_path=response.output_path,
                bucket=response.bucket,
                urls=response.urls,
            )

        except grpc.RpcError as e:
            raise QwakException(
                f"Failed to get pre signed urls for execution, error is: {e}"
            )

    def get_pre_signed_download_urls_details(
        self, execution_id: str
    ) -> GetBatchJobPreSignedDownloadUrlResult:
        """
        Get pre signed urls links in order to download files using Qwak cloud bucket
        Args:
            execution_id: The execution id of the files

        Returns: GetBatchJobPreSignedDownloadUrlResult which contains the pre-signed urls of the output files

        """
        try:
            response: GetBatchJobPreSignedDownloadUrlResponse = (
                self.batch_job_management.GetBatchJobPreSignedDownloadUrl(
                    GetBatchJobPreSignedDownloadUrlRequest(
                        job_id=execution_id,
                    ),
                    timeout=CLIENT_TIMEOUT,
                )
            )

            return GetBatchJobPreSignedDownloadUrlResult(
                success=response.success,
                failure_message=response.failure_message,
                urls=response.urls,
            )

        except grpc.RpcError as e:
            raise QwakException(
                f"Failed to get pre signed urls for execution, error is: {e}"
            )

    def get_upload_details(self, model_id: str) -> GetBatchJobUploadDetailsResponse:
        """
        Get upload details in order to start a job using Qwak cloud bucket
        Args:
        model_id: The model id for the execution

        Returns: GetBatchJobPreSignedUploadUrlResult which contains the input/output path, the bucket, and temporary credentials
        """
        try:
            return self.batch_job_management.GetBatchJobUploadDetails(
                GetBatchJobUploadDetailsRequest(
                    model_id=model_id,
                ),
                timeout=CLIENT_TIMEOUT,
            )

        except grpc.RpcError as e:
            raise QwakException(f"Failed to get upload details, error is: {e}")

    def get_download_details(
        self, execution_id: str
    ) -> GetBatchJobDownloadDetailsResponse:
        """
        Get download details in order to download files using Qwak cloud bucket
        Args:
        execution_id: The execution id of the files

        Returns: GetBatchJobDownloadDetailsResponse which contains the keys the bucket, and temporary credentials
        """
        try:
            return self.batch_job_management.GetBatchJobDownloadDetails(
                GetBatchJobDownloadDetailsRequest(
                    job_id=execution_id,
                ),
                timeout=CLIENT_TIMEOUT,
            )

        except grpc.RpcError as e:
            raise QwakException(f"Failed to get download details error is: {e}")
