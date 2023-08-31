import logging
import math
import os
import random
import string
import time
from io import BytesIO
from typing import List, Optional, Union

import pandas as pd
import requests
from _qwak_proto.qwak.batch_job.v1.batch_job_service_pb2 import (
    GetBatchJobUploadDetailsResponse,
)
from joblib import Parallel, delayed
from qwak.clients.batch_job_management import BatchJobManagerClient
from qwak.clients.batch_job_management.executions_config import ExecutionConfig
from qwak.clients.batch_job_management.results import (
    GetBatchJobPreSignedUploadUrlResult,
    StartExecutionResult,
    StartWarmupJobResult,
)
from qwak.clients.instance_template.client import InstanceTemplateManagementClient
from qwak.clients.secret_service.client import SecretServiceClient
from qwak.exceptions import QwakException

from qwak_inference.batch_client.file_serialization import (
    SERIALIZATION_HANDLER_MAP,
    SerializationFormat,
)
from qwak_inference.batch_client.instance_validation import verify_template_id
from qwak_inference.batch_client.s3 import S3Utils


class Color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GREY = "\033[98m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


class BatchInferenceClient:
    INTERMEDIATE_STATES = (
        "BATCH_JOB_COMMITTED_STATUS",
        "BATCH_JOB_PENDING_STATUS",
        "BATCH_JOB_RUNNING_STATUS",
    )
    FAILURE_STATES = (
        "BATCH_JOB_FAILED_STATUS",
        "BATCH_JOB_CANCELLED_STATUS",
        "BATCH_JOB_TIMEOUT_STATUS",
        "UNDEFINED_BATCH_JOB_STATUS",
    )
    SUCCESS_STATES = ("BATCH_JOB_FINISHED_STATUS",)

    MAX_UPLOAD_TRIES = 3
    MAX_DOWNLOAD_TRIES = 3
    MAX_PRE_SIGNED_BATCHES = 500

    def __init__(
        self,
        access_key_secret: Optional[str] = None,
        secret_access_key_secret: Optional[str] = None,
        bucket: Optional[str] = None,
        model_id: str = os.environ.get("QWAK_MODEL_ID"),
        log_level: int = logging.INFO,
    ):
        """Construct batch client for performing batch inference.

        Args:
            access_key_secret: AWS access key secret
            secret_access_key_secret: AWS secret access key secret
            bucket: AWS S3 bucket which will be used for storing the inference batch request and result.
            model_id: The model id that you want to perform  batch inference against.
            log_level: Logging level, Use logging level from std logging library.
        """
        if access_key_secret:
            logging.warning(
                f"{Color.RED}Passing `access_key_secret` will be deprecated. To stop passing it please "
                f"re-build your model with a qwak-sdk version >= 0.8.156.{Color.END}"
            )
        if secret_access_key_secret:
            logging.warning(
                f"{Color.RED}Passing `secret_access_key_secret` will be deprecated. To stop passing it please "
                f"re-build your model with a qwak-sdk version >= 0.8.156.{Color.END}"
            )
        if bucket:
            logging.warning(
                f"{Color.RED}Passing `bucket`  will be deprecated. To stop passing it please "
                f"re-build your model with a qwak-sdk version >= 0.8.156.{Color.END}"
            )
        self.model_id = model_id
        self.bucket = bucket
        self.access_key_secret = access_key_secret
        self.secret_access_key_secret = secret_access_key_secret
        self.s3_client = self._get_s3_client()
        self.batch_job_manager_client = BatchJobManagerClient()
        self.instance_template_client = InstanceTemplateManagementClient()

        logging.basicConfig(level=log_level)

    def _get_s3_client(self):
        if self.access_key_secret:
            secret_service_client = SecretServiceClient()
            aws_access_key_id = secret_service_client.get_secret(self.access_key_secret)
            aws_secret_access_key = secret_service_client.get_secret(
                self.secret_access_key_secret
            )
            return S3Utils.get_client(aws_access_key_id, aws_secret_access_key)
        else:
            return None

    def warm_up(
        self,
        executors: int = None,
        cpus: int = 0,
        memory: int = None,
        gpus: int = 0,
        gpu_type: str = None,
        timeout: int = None,
        build_id: str = None,
        instance: str = "",
    ) -> None:
        """Warmup instances for performing batch inference.

        Args:
            executors: Number of executors to warmup up.
            cpus: Requested amount of CPUs for each executor.
            memory: Requested amount of Memory for each executor.
            gpus: Requested amount of GPUs for each executor.
            gpu_type: Requested GPU type.
            timeout: Timeout for warmup.
            build_id: Build ID which will be used for performing the batch request.
            instance: The instance size to use in batch warmup. E.g.: 'small', 'medium', etc.
        """
        # Create qwak execution
        warmup_spec = ExecutionConfig.Warmup(timeout=timeout)
        execution_spec = ExecutionConfig.Execution(
            model_id=self.model_id, build_id=build_id
        )

        if instance:
            verify_template_id(instance, self.instance_template_client)

        resources_config = ExecutionConfig.Resources(
            pods=executors,
            cpus=cpus,
            memory=memory,
            gpu_type=gpu_type,
            gpu_amount=gpus,
            instance_size=instance,
        )
        execution_config = ExecutionConfig(
            warmup=warmup_spec, resources=resources_config, execution=execution_spec
        )
        warmup_result: StartWarmupJobResult = (
            self.batch_job_manager_client.start_warmup_job(execution_config)
        )
        if not warmup_result.success:
            error_message = (
                f"{Color.RED}An error occurred while starting warmup: {warmup_result.failure_message} "
                f"{Color.END}"
            )
            logging.error(error_message)
            raise QwakException(error_message)
        logging.info(f"{Color.GREEN}Started warmup{Color.END}")

    def run(
        self,
        df: pd.DataFrame,
        batch_size: int,
        job_timeout: int = 0,
        task_timeout: int = 0,
        executors: int = None,
        cpus: float = None,
        memory: int = None,
        gpus: int = 0,
        gpu_type: str = None,
        iam_role_arn: str = None,
        build_id: str = None,
        parameters: dict = None,
        serialization_format="PARQUET",
        instance: str = "",
    ) -> pd.DataFrame:
        """Perform batch inference on given data.

        Args:
            df: Dataframe to perform batch inference on.
            batch_size: Inference batch size.
            job_timeout: The entire execution job timeout.
            task_timeout: Timeout for processing a single file.
            executors: Number of executors to use in parallel.
            cpus: Requested amount of CPUs for each executor.
            memory: Requested amount of Memory for each executor.
            gpus: Requested amount of GPUs for each executor.
            gpu_type: Requested GPU type.
            iam_role_arn: IAM role arn to be used in executors.
            build_id: Build ID which will be used for performing the batch request.
            parameters:
            serialization_format: The format the files are saved in (Parquet, Feather, CSV)
            instance: The instance size to use in batch inference. E.g.: 'small', 'medium', etc.

        Returns:
            Dataframe inference result.
        """

        if serialization_format.upper() not in SERIALIZATION_HANDLER_MAP:
            raise ValueError(
                f"Invalid serialization format {serialization_format}."
                f" Supported types are: {list(SERIALIZATION_HANDLER_MAP.keys())}"
            )

        if instance:
            verify_template_id(instance, self.instance_template_client)

        serde_handler = SERIALIZATION_HANDLER_MAP.get(serialization_format.upper())

        destination_folder = None
        if self.s3_client:
            execution_spec, destination_folder = self.__client_creds_execute(
                df, batch_size, serde_handler
            )
        else:
            execution_spec = self.__default_auth_execute(df, batch_size, serde_handler)

        execution_spec.build_id = build_id
        if parameters is not None:
            execution_spec.parameters = parameters

        execution_spec.input_file_type = serde_handler.format_key
        execution_spec.output_file_type = serde_handler.format_key
        execution_spec.job_timeout = job_timeout
        execution_spec.file_timeout = task_timeout

        advanced_options_config = ExecutionConfig.AdvancedOptions(
            custom_iam_role_arn=iam_role_arn
        )

        resources_config = ExecutionConfig.Resources(
            pods=executors,
            cpus=cpus,
            memory=memory,
            gpu_type=gpu_type,
            gpu_amount=gpus,
            instance_size=instance,
        )
        execution_config = ExecutionConfig(
            execution=execution_spec,
            resources=resources_config,
            advanced_options=advanced_options_config,
        )

        execution_result: StartExecutionResult = (
            self.batch_job_manager_client.start_execution(execution_config)
        )
        if not execution_result.success:
            error_message = (
                f"{Color.RED}An error occurred while starting execution: "
                f"{execution_result.failure_message} {execution_result.execution_id}{Color.END}"
            )
            logging.error(error_message)
            raise QwakException(error_message)

        execution_id = execution_result.execution_id

        try:
            logging.info(f"{Color.GREEN}Started execution {execution_id}{Color.END}")

            job_status = self.get_job_status(execution_id)

            if job_status == "Successful":
                if self.s3_client:
                    return_df = self.__get_df_client_creds(
                        destination_folder, execution_id, serde_handler
                    )
                else:
                    return_df = self.__get_df_default_auth(execution_id, serde_handler)
                return return_df
            else:
                error_message = f"{Color.RED}Execution failed!{Color.END}"
                logging.error(error_message)
                raise QwakException(error_message)
        except KeyboardInterrupt:
            interrupted_msg = "Keyboard interrupted - canceling execution job"
            logging.warning(interrupted_msg)
            self.batch_job_manager_client.cancel_execution(execution_id)

            raise QwakException(interrupted_msg)

    def local_file_run(
        self,
        model_id: str,
        source_folder: str,
        destination_folder: str,
        input_file_type: str,
        output_file_type: str = None,
        job_timeout: int = 0,
        task_timeout: int = 0,
        executors: int = None,
        cpus: float = None,
        memory: int = None,
        gpus: int = 0,
        gpu_type: str = None,
        iam_role_arn: str = None,
        build_id: str = None,
        parameters: dict = None,
        instance: str = "",
    ) -> None:
        """Perform batch inference on given input directory (or a single input file), and store the result in the destination folder.

        Args:
            model_id: Model ID to perform batch inference on.
            source_folder: Input directory (or a single input file) to perform batch inference on.
            destination_folder: Output directory to store the result in. (will be created if it doesn't exist)
            job_timeout: The entire execution job timeout.
            task_timeout: Timeout for processing a single file.
            executors: Number of executors to use in parallel.
            cpus: Requested amount of CPUs for each executor.
            memory: Requested amount of Memory for each executor.
            gpus: Requested amount of GPUs for each executor.
            gpu_type: Requested GPU type.
            iam_role_arn: IAM role arn to be used in executors.
            build_id: Build ID which will be used for performing the batch request.
            parameters:
            instance: The instance size to use in batch inference. E.g.: 'small', 'medium', etc.

        Returns:
            None.
        """
        file_extension = input_file_type.lower() if input_file_type else "csv"
        destination_folder = destination_folder.removeprefix("file://")
        source_folder = source_folder.removeprefix("file://")
        if os.path.isdir(source_folder):
            files = [
                os.path.join(source_folder, f)
                for f in os.listdir(source_folder)
                if f.endswith(file_extension)
            ]
        else:
            files = [source_folder] if source_folder.endswith(file_extension) else []
        if files:
            upload_details = self._upload_input_files(model_id, files)

            execution_spec = ExecutionConfig.Execution(
                model_id=model_id,
                bucket=upload_details.bucket,
                source_folder=upload_details.input_path,
                destination_folder=upload_details.output_path.replace("//", "/"),
            )
            execution_spec.build_id = build_id
            if parameters is not None:
                execution_spec.parameters = parameters

            execution_spec.input_file_type = input_file_type
            execution_spec.output_file_type = (
                output_file_type if output_file_type else input_file_type
            )
            execution_spec.job_timeout = job_timeout
            execution_spec.file_timeout = task_timeout

            advanced_options_config = ExecutionConfig.AdvancedOptions(
                custom_iam_role_arn=iam_role_arn
            )

            resources_config = ExecutionConfig.Resources(
                pods=executors,
                cpus=cpus,
                memory=memory,
                gpu_type=gpu_type,
                gpu_amount=gpus,
                instance_size=instance,
            )
            execution_config = ExecutionConfig(
                execution=execution_spec,
                resources=resources_config,
                advanced_options=advanced_options_config,
            )

            execution_result: StartExecutionResult = (
                self.batch_job_manager_client.start_execution(execution_config)
            )
            if not execution_result.success:
                error_message = (
                    f"{Color.RED}An error occurred while starting execution: "
                    f"{execution_result.failure_message} {execution_result.execution_id}{Color.END}"
                )
                logging.error(error_message)
                raise QwakException(error_message)

            execution_id = execution_result.execution_id

            try:
                logging.info(
                    f"{Color.GREEN}Started execution {execution_id}{Color.END}"
                )
                job_status = self.get_job_status(execution_id)

                if job_status == "Successful":
                    self._download_batch_job_results_as_files(
                        execution_id, destination_folder
                    )
                else:
                    error_message = f"{Color.RED}Execution failed!{Color.END}"
                    logging.error(error_message)
                    raise QwakException(error_message)
            except KeyboardInterrupt:
                interrupted_msg = "Keyboard interrupted - canceling execution job"
                logging.warning(interrupted_msg)
                self.batch_job_manager_client.cancel_execution(execution_id)

                raise QwakException(interrupted_msg)
        else:
            raise ValueError(
                f"No files found in {source_folder} with extension {file_extension}"
            )

    def _upload_input_files(self, model_id, files):
        upload_details: GetBatchJobUploadDetailsResponse = (
            self.batch_job_manager_client.get_upload_details(model_id)
        )
        s3_client = S3Utils.get_client_with_temp_creds(upload_details.credentials)

        for file in files:
            file_name = os.path.basename(file)
            s3_client.upload_file(
                file, upload_details.bucket, f"{upload_details.input_path}/{file_name}"
            )

        return upload_details

    def _download_batch_job_results_as_files(self, execution_id, destination_folder):
        download_details = self.batch_job_manager_client.get_download_details(
            execution_id
        )
        s3_client = S3Utils.get_client_with_temp_creds(download_details.credentials)

        os.makedirs(destination_folder, exist_ok=True)
        for key in download_details.keys:
            file_name = os.path.basename(key)
            s3_client.download_file(
                download_details.bucket, key, f"{destination_folder}/{file_name}"
            )

    def __default_auth_execute(
        self, df: pd.DataFrame, batch_size: int, serde_handler: SerializationFormat
    ):
        num_of_batches = math.ceil(len(df.index) / batch_size)
        if num_of_batches > self.MAX_PRE_SIGNED_BATCHES:
            raise QwakException(
                f"Number of batches is larger than {self.MAX_PRE_SIGNED_BATCHES} ({num_of_batches}). "
                f"Please decrease the number of batches. You can do this by increasing the `batch_size` parameter "
                f"in the execution."
            )
        pre_signed_results: GetBatchJobPreSignedUploadUrlResult = (
            self.batch_job_manager_client.get_pre_signed_upload_urls_details(
                self.model_id,
                num_of_batches,
                file_type=serde_handler.get_file_type(),
            )
        )

        if not pre_signed_results.success:
            raise QwakException(
                "Couldn't create pre signed urls in order to upload DataFrame to. "
                f"Error is: {pre_signed_results.failure_message}"
            )

        source_folder = pre_signed_results.input_path
        destination_folder = pre_signed_results.output_path
        self.bucket = pre_signed_results.bucket

        # Upload files to S3
        self._upload_df(
            df=df,
            batch_size=batch_size,
            pre_signed_urls=pre_signed_results.urls,
            serde_handler=serde_handler,
        )

        # Create qwak execution
        return ExecutionConfig.Execution(
            model_id=self.model_id,
            bucket=self.bucket,
            source_folder=source_folder,
            destination_folder=destination_folder,
        )

    def __client_creds_execute(
        self, df: pd.DataFrame, batch_size: int, serde_handler: SerializationFormat
    ):
        job_id = "".join(
            random.choices(string.ascii_letters + string.digits, k=16)  # nosec
        )

        # Upload files to S3
        source_folder = self._upload_df(
            df=df,
            batch_size=batch_size,
            pre_signed_urls=None,
            job_id=job_id,
            serde_handler=serde_handler,
        )

        destination_folder = f"qwak/{self.model_id}/{job_id}/output"

        # Create qwak execution
        return (
            ExecutionConfig.Execution(
                model_id=self.model_id,
                bucket=self.bucket,
                source_folder=source_folder,
                input_file_type=serde_handler.format_key,
                destination_folder=destination_folder,
                access_token_name=self.access_key_secret,
                access_secret_name=self.secret_access_key_secret,
            ),
            destination_folder,
        )

    def cancel(self, execution_id):
        """

        Returns:

        """
        status_response = self.batch_job_manager_client.cancel_execution(
            execution_id=execution_id
        )
        if not status_response.success:
            logging.error(
                f"{Color.RED}Failed to cancel job {status_response.failure_message}{Color.END}"
            )
        else:
            logging.info(f"{Color.GREEN}Execution {execution_id} canceled{Color.END}")

    def get_job_status(self, execution_id):
        """

        Args:
            execution_id:

        Returns:

        """
        errors = 0
        while True:
            time.sleep(2)
            try:
                status_response = self.batch_job_manager_client.get_execution_status(
                    execution_id
                )
            except QwakException:
                continue
            if not status_response.success:
                logging.error(
                    f"{Color.RED}Failed to get status {status_response.failure_message}{Color.END}"
                )
                if errors < 3:
                    errors += 1
                    continue
                else:
                    logging.error(
                        f"{Color.RED}Failed to get status 3 times. Quitting{Color.END}"
                    )
                    break
            errors = 0
            status = status_response.status
            if status in self.INTERMEDIATE_STATES:
                logging.info(
                    f"{Color.BLUE}Current status is: {status} {status_response.finished_files}/"
                    f"{status_response.total_files}{Color.END}"
                )
            elif status in self.SUCCESS_STATES:
                logging.info(f"{Color.GREEN}Execution finished successfully{Color.END}")
                return "Successful"
            elif status in self.FAILURE_STATES:
                execution_report = self.batch_job_manager_client.get_execution_report(
                    execution_id
                )
                if execution_report.success:
                    report = "\n".join(execution_report.records)
                    logs = "\n".join(execution_report.model_logs)
                    logging.error(f"{Color.WHITE}{report}\n{logs}{Color.END}")
                return "Failed"

    def _upload_df(
        self,
        df: pd.DataFrame,
        batch_size: int,
        serde_handler: SerializationFormat,
        pre_signed_urls: Union[List[str], None],
        job_id: str = None,
    ):
        """

        Args:
            df: Dataframe
            batch_size: Number of rows in each batch
            serde_handler: Serializer/Deserializer handler
            pre_signed_urls: List of pre signed urls to upload the batches to
            job_id: Batch job Id

        Returns:

        """

        def upload_chunked_df_client_creds(i: int, df_chunk: pd.DataFrame):
            self.s3_client.put_object(
                Body=serde_handler.get_bytes(df_chunk),
                Bucket=self.bucket,
                Key=f"qwak/{self.model_id}/{job_id}/input/split_{i}.{serde_handler.get_file_type()}",
            )

        def upload_chunked_df_default_auth(i: int, df_chunk: pd.DataFrame):
            tries = 0
            while tries < self.MAX_UPLOAD_TRIES:
                logging.info(f"Trying to upload chunk {i}, attempt #{tries}")
                result = requests.put(
                    pre_signed_urls[i],
                    data=serde_handler.get_bytes(df_chunk),
                    headers={"Content-Type": "application/binary"},
                    timeout=300,
                )
                if result.status_code == 200:
                    return
                else:
                    logging.error(
                        f"Failed uploading chunk {i}. Details are: {result.status_code} : "
                        f"{result.reason} : {result.content}"
                    )
                    tries += 1
            else:
                raise QwakException(
                    f"An error occurred while sending the data: {result.reason}"
                )

        def batch(iterable, n=1):
            length = iterable.shape[0]
            for ndx in range(0, length, n):
                yield iterable[ndx : min(ndx + n, length)]

        # Upload df parts in parallel according to # of CPU's
        upload_func = (
            upload_chunked_df_client_creds
            if self.s3_client
            else upload_chunked_df_default_auth
        )
        Parallel(n_jobs=-1, prefer="threads")(
            delayed(upload_func)(i, x) for i, x in enumerate(batch(df, batch_size))
        )

        return f"qwak/{self.model_id}/{job_id}/input"

    def __get_df_client_creds(
        self, prefix: str, execution_id: str, serde_handler: SerializationFormat
    ):
        objects = self.s3_client.list_objects(
            Bucket=self.bucket,
            Prefix=f"{prefix}/{execution_id}/results",
        )
        dfs = []
        for object in objects["Contents"]:
            response = self.s3_client.get_object(
                Bucket=self.bucket, Key=f"{object['Key']}"
            )
            dfs.append(
                serde_handler.read_df(
                    BytesIO(b"".join(response.get("Body").readlines()))
                )
            )
        return pd.concat(dfs, axis=0, ignore_index=True)

    def __get_df_default_auth(
        self, execution_id: str, serde_handler: SerializationFormat
    ) -> pd.DataFrame:
        pre_signed_download_urls = (
            self.batch_job_manager_client.get_pre_signed_download_urls_details(
                execution_id
            ).urls
        )
        dfs = []
        for url in pre_signed_download_urls:
            tries = 0
            while tries < self.MAX_DOWNLOAD_TRIES:
                response = requests.get(url, timeout=300)
                if response.status_code == 200:
                    dfs.append(serde_handler.read_df(BytesIO(response.content)))
                    break
                else:
                    tries += 1
                    logging.warning(
                        f"Failed downloading results file [{response.status_code}]: {response.reason}"
                    )
            else:
                raise QwakException(
                    f"An error occurred while downloading the data: {response.reason}"
                )

        return pd.concat(dfs, axis=0).sort_index()
