import grpc
from _qwak_proto.qwak.feature_store.jobs.v1.job_pb2 import JobRecord, JobState
from _qwak_proto.qwak.feature_store.jobs.v1.job_service_pb2 import (
    ApplyJobRequest,
    ApplyJobResponse,
    GetLatestJobRequest,
    GetLatestJobResponse,
    GetLatestSuccessfulJobRequest,
    GetLatestSuccessfulJobResponse,
)
from _qwak_proto.qwak.feature_store.jobs.v1.job_service_pb2_grpc import (
    JobServiceServicer,
)
from qwak_services_mock.mocks.utils.exception_handlers import raise_internal_grpc_error


def _sort_by_running_id(job: JobRecord):
    return job.run_id


class JobRegistryServiceApiMock(JobServiceServicer):
    def __init__(self):
        self._job_records: dict[str:[JobRecord]] = {}
        self._running_id = 0
        super(JobRegistryServiceApiMock, self).__init__()

    def given_job(self, feature_set_id, job_id, job_state):
        new_job = JobRecord(
            job_id=job_id, featureset_id=feature_set_id, job_state=job_state
        )
        self._running_id += 1
        new_job.run_id = self._running_id

        self._job_records[new_job.featureset_id] = self._job_records.get(
            new_job.featureset_id, list()
        ) + [new_job]

    def GetLatestJob(self, request: GetLatestJobRequest, context):
        """
        Get latest job
        """
        try:
            records: list[JobRecord] = self._job_records.get(request.featureset_id)
            if records:
                records.sort(key=_sort_by_running_id, reverse=True)
                return GetLatestJobResponse(job=records[0])
            else:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("Received a non existing job record")
                return GetLatestSuccessfulJobResponse()
        except Exception as e:
            raise_internal_grpc_error(context, e)

    def GetLatestSuccessfulJob(self, request: GetLatestSuccessfulJobRequest, context):
        """
        Get latest successful job
        """
        try:
            records: list[JobRecord] = self._job_records.get(request.featureset_id)
            if records:
                records.sort(key=_sort_by_running_id, reverse=True)
                for job in records:
                    if job.job_state == JobState.COMPLETED:
                        return GetLatestSuccessfulJobResponse(job=job)
                return GetLatestSuccessfulJobResponse()
            else:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("Received a non existing feature job record")
                return GetLatestSuccessfulJobResponse()
        except Exception as e:
            raise_internal_grpc_error(context, e)
