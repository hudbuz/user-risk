from qwak.clients.batch_job_management import BatchJobManagerClient
from qwak.clients.batch_job_management.results import CancelExecutionResult


def execute_execution_cancel(execution_id: str):
    batch_job_cancel_response: CancelExecutionResult = (
        BatchJobManagerClient().cancel_execution(execution_id)
    )
    return batch_job_cancel_response.success, batch_job_cancel_response.failure_message
