from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

from _qwak_proto.qwak.batch_job.v1.batch_job_resources_pb2 import (
    NVIDIA_A10G,
    NVIDIA_A100,
    NVIDIA_K80,
    NVIDIA_T4,
    NVIDIA_V100,
)
from _qwak_proto.qwak.batch_job.v1.batch_job_service_pb2 import (
    CSV_INPUT_FILE_TYPE,
    CSV_OUTPUT_FILE_TYPE,
    FEATHER_INPUT_FILE_TYPE,
    FEATHER_OUTPUT_FILE_TYPE,
    PARQUET_INPUT_FILE_TYPE,
    PARQUET_OUTPUT_FILE_TYPE,
)
from qwak.exceptions import QwakException
from qwak.inner.tool.run_config import (
    ConfigCliMap,
    QwakConfigBase,
    YamlConfigMixin,
    validate_float,
    validate_int,
    validate_list_of_strings,
    validate_string,
)

DEFAULT_OUTPUT_FORMAT = "CSV"
OUTPUT_FORMATTERS_MAP = {
    DEFAULT_OUTPUT_FORMAT: CSV_OUTPUT_FILE_TYPE,
    "PARQUET": PARQUET_OUTPUT_FILE_TYPE,
    "FEATHER": FEATHER_OUTPUT_FILE_TYPE,
}

DEFAULT_INPUT_FORMAT = "CSV"
INPUT_FORMATTERS_MAP = {
    DEFAULT_INPUT_FORMAT: CSV_INPUT_FILE_TYPE,
    "PARQUET": PARQUET_INPUT_FILE_TYPE,
    "FEATHER": FEATHER_INPUT_FILE_TYPE,
}

GPU_TYPE_MAP = {
    "NVIDIA_V100": NVIDIA_V100,
    "NVIDIA_T4": NVIDIA_T4,
    "NVIDIA_K80": NVIDIA_K80,
    "NVIDIA_A100": NVIDIA_A100,
    "NVIDIA_A10G": NVIDIA_A10G,
}

CONFIG_MAPPING: List[ConfigCliMap] = [
    ConfigCliMap("model_id", "execution.model_id", validate_string, True),
    ConfigCliMap("build_id", "execution.build_id", validate_string, False),
    ConfigCliMap("branch", "execution.branch", validate_string),
    ConfigCliMap("source_bucket", "execution.source_bucket", validate_string, False),
    ConfigCliMap("source_folder", "execution.source_folder", validate_string, False),
    ConfigCliMap("bucket", "execution.bucket", validate_string, False),
    ConfigCliMap(
        "destination_bucket", "execution.destination_bucket", validate_string, False
    ),
    ConfigCliMap(
        "destination_folder", "execution.destination_folder", validate_string, False
    ),
    ConfigCliMap(
        "input_file_type",
        "execution.input_file_type",
        validate_string,
        False,
    ),
    ConfigCliMap(
        "output_file_type",
        "execution.output_file_type",
        validate_string,
        False,
    ),
    ConfigCliMap("param_list", "execution.parameters", validate_list_of_strings, False),
    ConfigCliMap("job_timeout", "execution.job_timeout", validate_int, False),
    ConfigCliMap("file_timeout", "execution.file_timeout", validate_int, False),
    ConfigCliMap(
        "access_token_name", "execution.access_token_name", validate_string, False
    ),
    ConfigCliMap(
        "access_secret_name", "execution.access_secret_name", validate_string, False
    ),
    ConfigCliMap("pods", "resources.pods", validate_int, False),
    ConfigCliMap("cpus", "resources.cpus", validate_float, False),
    ConfigCliMap("memory", "resources.memory", validate_int, False),
    ConfigCliMap("gpu_type", "resources.gpu_type", validate_string, False),
    ConfigCliMap("gpu_amount", "resources.gpu_amount", validate_int, False),
    ConfigCliMap("instance", "resources.instance_size", validate_string, False),
]


@dataclass
class ExecutionConfig(YamlConfigMixin, QwakConfigBase):
    def merge_bucket_arguments(self):
        if self.execution.bucket:
            if not self.execution.source_bucket:
                self.execution.source_bucket = self.execution.bucket
            if not self.execution.destination_bucket:
                self.execution.destination_bucket = self.execution.bucket

    def __post_init__(self):
        self.merge_bucket_arguments()

    def _post_merge_cli(self):
        self.merge_bucket_arguments()
        self.execution.parameters = dictify_params(self.execution.parameters)

        if not self.execution.source_bucket:
            raise QwakException(
                "Must supply either --bucket or --source-bucket parameter"
            )
        if not self.execution.destination_bucket:
            raise QwakException(
                "Must supply either --bucket or --destination-bucket parameter"
            )

    @property
    def _config_mapping(self) -> List[ConfigCliMap]:
        return CONFIG_MAPPING

    @dataclass
    class Execution:
        model_id: str = field(default="")
        build_id: str = field(default="")
        branch: str = field(default="main")
        bucket: str = field(default="")
        source_bucket: str = field(default="")
        source_folder: str = field(default="")
        destination_bucket: str = field(default="")
        destination_folder: str = field(default="")
        input_file_type: str = field(default=DEFAULT_INPUT_FORMAT)
        output_file_type: str = field(default=DEFAULT_OUTPUT_FORMAT)
        access_token_name: str = field(default="")
        access_secret_name: str = field(default="")
        job_timeout: int = field(default=0)
        file_timeout: int = field(default=0)
        parameters: dict = field(default_factory=lambda: defaultdict(str))

    @dataclass
    class Warmup:
        model_id: str = field(default="")
        branch: str = field(default="main")
        timeout: int = field(default=60 * 60)

    @dataclass
    class Resources:
        pods: int = field(default=0)
        cpus: float = field(default=0)
        memory: int = field(default=0)
        gpu_amount: int = field(default=0)
        gpu_type: str = field(default="")
        instance_size: str = field(default="")

    @dataclass
    class AdvancedOptions:
        custom_iam_role_arn: str = field(default=None)

    warmup: Warmup = field(default_factory=Warmup)
    execution: Execution = field(default_factory=Execution)
    resources: Resources = field(default_factory=Resources)
    advanced_options: AdvancedOptions = field(default_factory=AdvancedOptions)


def dictify_params(parameters: Union[Dict[str, str], List[str], Tuple[str]]):
    if isinstance(parameters, dict):
        return parameters

    result = dict()

    if isinstance(parameters, (list, tuple)):
        for param in parameters:
            if "=" not in param:
                raise QwakException(
                    f'The parameter definition passed {param} is invalid. Format is "KEY=VALUE"'
                )
            split_param = param.split("=")
            result[split_param[0]] = split_param[1]

    return result
