from qwak.clients.batch_job_management import BatchJobManagerClient
from qwak.clients.batch_job_management.results import ExecutionStatusResult


def execute_execution_status(execution_id: str):
    batch_job_status_response: ExecutionStatusResult = (
        BatchJobManagerClient().get_execution_status(execution_id)
    )
    return (
        batch_job_status_response.status,
        batch_job_status_response.success,
        batch_job_status_response.failure_message,
    )
