from typing import Tuple

from qwak.clients.batch_job_management import BatchJobManagerClient
from qwak.clients.batch_job_management.executions_config import ExecutionConfig
from qwak.clients.batch_job_management.results import StartExecutionResult
from qwak.clients.instance_template.client import InstanceTemplateManagementClient
from qwak.tools.logger.logger import get_qwak_logger

from qwak_sdk.commands._logic.instance_template import verify_template_id

logger = get_qwak_logger()


def execute_start_execution(config: ExecutionConfig) -> Tuple[str, bool, str]:
    if config.resources.instance_size:
        verify_template_id(
            config.resources.instance_size, InstanceTemplateManagementClient()
        )
    batch_job_start_response: StartExecutionResult = (
        BatchJobManagerClient().start_execution(config)
    )
    return (
        batch_job_start_response.execution_id,
        batch_job_start_response.success,
        batch_job_start_response.failure_message,
    )


def execute_in_local_mode(kwargs):
    from qwak_inference.batch_client.batch_client import BatchInferenceClient

    logger.info("Executing a batch run in local file mode...")

    model_id = kwargs.get("model_id")
    source_folder = kwargs.get("source_folder")
    destination_folder = kwargs.get("destination_folder")
    input_file_type = kwargs.get("input_file_type")
    output_file_type = kwargs.get("output_file_type", None)
    job_timeout = kwargs.get("job_timeout", 0)
    task_timeout = kwargs.get("task_timeout", 0)
    executors = kwargs.get("executors", None)
    cpus = kwargs.get("cpus", None)
    memory = kwargs.get("memory", None)
    gpus = kwargs.get("gpus", 0)
    gpu_type = kwargs.get("gpu_type", None)
    iam_role_arn = kwargs.get("iam_role_arn", None)
    build_id = kwargs.get("build_id", None)
    parameters = kwargs.get("parameters", None)
    instance = kwargs.get("instance", "")

    if (
        not model_id
        or not source_folder
        or not destination_folder
        or not input_file_type
    ):
        raise ValueError(
            "model-id, source-folder, destination-folder and input-file-type are required"
        )

    client = BatchInferenceClient()
    client.local_file_run(
        model_id=model_id,
        source_folder=source_folder,
        destination_folder=destination_folder,
        input_file_type=input_file_type,
        output_file_type=output_file_type,
        job_timeout=job_timeout,
        task_timeout=task_timeout,
        executors=executors,
        cpus=cpus,
        memory=memory,
        gpus=gpus,
        gpu_type=gpu_type,
        iam_role_arn=iam_role_arn,
        build_id=build_id,
        parameters=parameters,
        instance=instance,
    )
    logger.info(
        "Batch run completed successfully. Results are available in the destination folder."
    )
