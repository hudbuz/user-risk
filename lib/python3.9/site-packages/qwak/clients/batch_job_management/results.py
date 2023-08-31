from dataclasses import dataclass, field
from typing import List


@dataclass
class StartExecutionResult:
    success: bool = field(default=True)
    execution_id: str = field(default="")
    failure_message: str = field(default="")


@dataclass
class StartWarmupResult:
    success: bool = field(default=True)
    failure_message: str = field(default="")


@dataclass
class ExecutionStatusResult:
    success: bool = field(default=True)
    failure_message: str = field(default="")
    status: str = field(default="")
    finished_files: int = field(default=0)
    total_files: int = field(default=0)


@dataclass
class CancelExecutionResult:
    success: bool = field(default=True)
    failure_message: str = field(default="")


@dataclass
class CancelWarmupResult:
    success: bool = field(default=True)
    failure_message: str = field(default="")


@dataclass
class GetExecutionReportResult:
    success: bool = field(default=True)
    failure_message: str = field(default="")
    records: List[str] = field(default_factory=list)
    model_logs: List[str] = field(default_factory=list)


@dataclass
class GetBatchJobPreSignedUploadUrlResult:
    success: bool = field(default=True)
    failure_message: str = field(default="")
    input_path: str = field(default="")
    output_path: str = field(default="")
    bucket: str = field(default="")
    urls: List[str] = field(default_factory=list)


@dataclass
class GetBatchJobPreSignedDownloadUrlResult:
    success: bool = field(default=True)
    failure_message: str = field(default="")
    urls: List[str] = field(default_factory=list)


@dataclass
class StartWarmupJobResult:
    success: bool = field(default=True)
    failure_message: str = field(default="")


@dataclass
class CancelWarmupJobResult:
    success: bool = field(default=True)
    failure_message: str = field(default="")
