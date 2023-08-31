from qwak.clients.batch_job_management import BatchJobManagerClient
from qwak.clients.batch_job_management.results import GetExecutionReportResult


def execute_execution_report(execution_id: str):
    batch_job_report_response: GetExecutionReportResult = (
        BatchJobManagerClient().get_execution_report(execution_id)
    )
    return (
        batch_job_report_response.records,
        batch_job_report_response.model_logs,
        batch_job_report_response.success,
        batch_job_report_response.failure_message,
    )
