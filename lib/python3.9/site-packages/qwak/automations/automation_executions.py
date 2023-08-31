from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from _qwak_proto.qwak.automation.v1.automation_execution_pb2 import (
    AutomationExecutionMessage,
    ExecutionRunDetailsMessage,
    ExecutionStatusMessage,
)
from google.protobuf.timestamp_pb2 import Timestamp


class ExecutionStatus(Enum):
    UNDEFINED = 0
    RUNNING = 1
    SUCCESSFUL = 2
    FAILED = 3


execution_status_to_proto_map = {
    ExecutionStatus.UNDEFINED: ExecutionStatusMessage.UNDEFINED,
    ExecutionStatus.RUNNING: ExecutionStatusMessage.RUNNING,
    ExecutionStatus.FAILED: ExecutionStatusMessage.FAILED,
    ExecutionStatus.SUCCESSFUL: ExecutionStatusMessage.SUCCESSFUL,
}

proto_status_to_execution_status_map = {
    v: k for k, v in execution_status_to_proto_map.items()
}


@dataclass
class ExecutionRunDetails:
    start_time: datetime = field(default=None)
    end_time: datetime = field(default=None)
    error_details: str = field(default="")
    status: ExecutionStatus = field(default=ExecutionStatus.UNDEFINED)
    task: str = field(default="")
    finish_cause: str = field(default="")

    def to_proto(self):
        start_time_timestamp = None
        end_time_timestamp = None
        if self.start_time:
            start_time_timestamp = Timestamp()
            start_time_timestamp.FromDatetime(self.start_time)

        if self.end_time:
            end_time_timestamp = Timestamp()
            end_time_timestamp.FromDatetime(self.end_time)

        return ExecutionRunDetailsMessage(
            start_time=start_time_timestamp,
            end_time=end_time_timestamp,
            error_details=self.error_details,
            status=execution_status_to_proto_map.get(self.status),
            task=self.task,
            finish_cause=self.finish_cause,
        )

    @staticmethod
    def from_proto(message: ExecutionRunDetailsMessage):
        return ExecutionRunDetails(
            start_time=datetime.fromtimestamp(
                message.start_time.seconds + message.start_time.nanos / 1e9
            )
            if message.start_time
            else None,
            end_time=datetime.fromtimestamp(
                message.end_time.seconds + message.end_time.nanos / 1e9
            )
            if message.end_time
            else None,
            status=proto_status_to_execution_status_map.get(message.status),
            task=message.task,
            error_details=message.error_details,
            finish_cause=message.finish_cause,
        )


@dataclass
class AutomationExecution:
    execution_id: str = field(default="")
    run_details: ExecutionRunDetails = field(default_factory=ExecutionRunDetails)

    @staticmethod
    def from_proto(message: AutomationExecutionMessage):
        return AutomationExecution(
            execution_id=message.execution_id,
            run_details=ExecutionRunDetails.from_proto(message.run_details),
        )

    def __str__(self):
        return f"Id: {self.execution_id}\nDetails:\nStart Time:\t{self.run_details.start_time}\nEnd Time:\t{self.run_details.end_time}\nStatus:\t{self.run_details.status.name}\n"
