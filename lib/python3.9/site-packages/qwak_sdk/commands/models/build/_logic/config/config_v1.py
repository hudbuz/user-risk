from dataclasses import dataclass, field
from os.path import expandvars
from typing import Dict, List, Optional, Tuple, Union

from _qwak_proto.qwak.builds.builds_pb2 import (
    BuildConfiguration,
    BuildEnvironment,
    BuildModelUri,
    BuildPropertiesProto,
    CondaBuild,
    DockerBuild,
    PoetryBuild,
    PythonBuild,
    RemoteBuild,
    RemoteBuildResources,
    Step,
    VirtualEnvironmentBuild,
)
from qwak.clients.build_orchestrator import BuildOrchestratorClient
from qwak.clients.instance_template.client import InstanceTemplateManagementClient
from _qwak_proto.qwak.builds.build_pb2 import BaseDockerImageType
from _qwak_proto.qwak.instance_template.instance_template_pb2 import InstanceType
from qwak.inner.tool.run_config import (
    QwakConfigBase,
    YamlConfigMixin,
    validate_bool,
    validate_float,
    validate_int,
    validate_list_of_strings,
    validate_string,
)

from qwak_sdk.commands.models.build._logic.util.protobuf_factory import protobuf_factory
from qwak_sdk.common.run_config import ConfigCliMap

CONFIG_MAPPING: List[ConfigCliMap] = [
    # ConfigV1
    ConfigCliMap(
        "validate_build_artifact", "step.validate_build_artifact", validate_bool, True
    ),
    ConfigCliMap(
        "validate_build_artifact_timeout",
        "step.validate_build_artifact_timeout",
        validate_int,
        True,
    ),
    ConfigCliMap("tests", "step.tests", validate_bool, True),
    # BuildProperties
    ConfigCliMap("model_id", "build_properties.model_id", validate_string, True),
    ConfigCliMap("build_id", "build_properties.build_id", validate_string, False),
    ConfigCliMap(
        "gpu_compatible", "build_properties.gpu_compatible", validate_bool, True
    ),
    ConfigCliMap("branch", "build_properties.branch", validate_string),
    ConfigCliMap("tags", "build_properties.tags", validate_list_of_strings),
    # ModelUri
    ConfigCliMap("uri", "build_properties.model_uri.uri", validate_string, False),
    ConfigCliMap(
        "git_branch", "build_properties.model_uri.git_branch", validate_string
    ),
    ConfigCliMap("main_dir", "build_properties.model_uri.main_dir", validate_string),
    ConfigCliMap(
        "git_credentials", "build_properties.model_uri.git_credentials", validate_string
    ),
    ConfigCliMap(
        "git_credentials_secret",
        "build_properties.model_uri.git_credentials_secret",
        validate_string,
    ),
    ConfigCliMap(
        "dependency_required_folders",
        "build_properties.model_uri.dependency_required_folders",
        validate_list_of_strings,
    ),
    # BuildEnv
    # PythonEnv
    ConfigCliMap(
        "dependency_file_path",
        "build_env.python_env.dependency_file_path",
        validate_string,
    ),
    # DockerConf
    ConfigCliMap("base_image", "build_env.docker.base_image", validate_string),
    ConfigCliMap("param_list", "build_env.docker.params", validate_list_of_strings),
    ConfigCliMap("env_vars", "build_env.docker.env_vars", validate_list_of_strings),
    ConfigCliMap("cache", "build_env.docker.cache", validate_bool),
    ConfigCliMap(
        "iam_role_arn", "build_env.docker.assumed_iam_role_arn", validate_string
    ),
    # RemoteConf
    ConfigCliMap("cpus", "build_env.remote.resources.cpus", validate_float),
    ConfigCliMap("memory", "build_env.remote.resources.memory", validate_string),
    ConfigCliMap("gpu_type", "build_env.remote.resources.gpu_type", validate_string),
    ConfigCliMap("gpu_amount", "build_env.remote.resources.gpu_amount", validate_int),
    ConfigCliMap("instance", "build_env.remote.resources.instance", validate_string),
    # Verbosity level
    ConfigCliMap("verbose", "verbose", validate_int),
    ConfigCliMap("deploy", "deploy", validate_bool),
    ConfigCliMap("deployment_instance", "deployment_instance", validate_string),
]


@protobuf_factory(BuildConfiguration, exclude_fields=["deploy"])
@dataclass
class ConfigV1(YamlConfigMixin, QwakConfigBase):
    @property
    def _config_mapping(self) -> List[ConfigCliMap]:
        return CONFIG_MAPPING

    @protobuf_factory(Step)
    @dataclass
    class Step:
        validate_build_artifact: bool = field(default=True)
        validate_build_artifact_timeout: int = field(default=120)
        tests: bool = field(default=True)

    @protobuf_factory(BuildPropertiesProto)
    @dataclass
    class BuildProperties:
        @protobuf_factory(
            BuildModelUri,
            exclude_fields=[
                "git_credentials",
                "git_credentials_secret",
                "dependency_required_folders",
            ],
        )
        @dataclass
        class ModelUri:
            uri: Optional[str] = field(default=None)
            git_branch: Optional[str] = field(default="master")
            main_dir: Optional[str] = field(default="main")
            dependency_required_folders: List[str] = field(default_factory=list)
            git_credentials: Optional[str] = field(default=None)
            git_credentials_secret: Optional[str] = field(default=None)

        build_id: Optional[str] = field(default=None)
        gpu_compatible: bool = field(default=False)
        branch: Optional[str] = field(default="main")
        model_id: str = field(default="")
        tags: List[str] = field(default_factory=list)
        model_uri: ModelUri = field(default_factory=ModelUri)

    @protobuf_factory(BuildEnvironment)
    @dataclass
    class BuildEnv:
        @protobuf_factory(DockerBuild)
        @dataclass
        class DockerConf:
            base_image: Optional[str] = field(default=None)
            env_vars: Optional[Union[List[str], Dict[str, str], Tuple]] = field(
                default_factory=list
            )
            assumed_iam_role_arn: str = field(default=None)
            params: Optional[List[str]] = field(default_factory=list)
            no_cache: bool = field(default=False)
            cache: bool = field(default=True)

        @protobuf_factory(
            PythonBuild, exclude_fields=["git_credentials_secret", "git_credentials"]
        )
        @dataclass
        class PythonEnv:
            @protobuf_factory(PoetryBuild)
            @dataclass
            class PoetryType:
                python_version: Optional[str] = field(default=None)
                lock_file: Optional[str] = field(default=None)

            @protobuf_factory(CondaBuild)
            @dataclass
            class CondaType:
                conda_file: Optional[str] = field(default=None)

            @protobuf_factory(VirtualEnvironmentBuild)
            @dataclass
            class VirtualenvType:
                python_version: Optional[str] = field(default=None)
                requirements_txt: Optional[str] = field(default=None)

            git_credentials: Optional[str] = field(default=None)
            git_credentials_secret: Optional[str] = field(default=None)
            poetry: PoetryType = field(default=None)
            conda: CondaType = field(default=None)
            virtualenv: VirtualenvType = field(default=None)
            dependency_file_path: Optional[str] = field(default=None)

            def __post_init__(self):
                if not (self.poetry or self.conda or self.virtualenv):
                    self.conda = self.CondaType("conda.yml")

        @protobuf_factory(RemoteBuild)
        @dataclass
        class RemoteConf:
            @protobuf_factory(RemoteBuildResources)
            @dataclass
            class RemoteBuildResources:
                cpus: Optional[float] = field(default=None)
                memory: Optional[str] = field(default=None)
                gpu_type: Optional[str] = field(default=None)
                gpu_amount: Optional[int] = field(default=None)
                instance: Optional[str] = field(default=None)

            is_remote: bool = field(default=True)
            resources: RemoteBuildResources = field(
                default_factory=RemoteBuildResources
            )

        docker: DockerConf = field(default_factory=DockerConf)
        python_env: PythonEnv = field(default_factory=PythonEnv)
        remote: RemoteConf = field(default_factory=RemoteConf)

    build_properties: BuildProperties = field(default_factory=BuildProperties)
    build_env: Optional[BuildEnv] = field(default_factory=BuildEnv)
    pre_build: Optional[BuildEnv] = field(default=None)
    post_build: Optional[BuildEnv] = field(default=None)
    step: Step = field(default_factory=Step)
    deploy: bool = field(default=False)
    deployment_instance: Optional[str] = field(default=None)
    verbose: int = field(default=0)

    def __post_init__(self):
        if isinstance(self.build_env.docker.env_vars, (list, tuple)):
            self.build_env.docker.env_vars = [
                expandvars(var) for var in self.build_env.docker.env_vars
            ]

    def _post_merge_cli(self):
        """Implementing post merge actions."""
        self.build_properties.model_id = self.build_properties.model_id.lower()
        self.build_properties.tags = list(self.build_properties.tags)
        self.build_properties.model_uri.dependency_required_folders = list(
            self.build_properties.model_uri.dependency_required_folders
        )
        self.build_env.docker.env_vars = list(self.build_env.docker.env_vars)
        self.build_env.docker.params = list(self.build_env.docker.params)

    def fetch_base_docker_image_name(self, build_orchestrator: BuildOrchestratorClient, instance_template: InstanceTemplateManagementClient):
        if not self.build_env.docker.base_image:
            is_gpu_used = self.build_env.remote.resources.gpu_type \
                          and self.build_env.remote.resources.gpu_amount \
                          and self.build_env.remote.resources.gpu_amount > 0

            if self.build_env.remote.resources.instance and not is_gpu_used:
                instance_spec = instance_template.get_instance_template(self.build_env.remote.resources.instance)
                is_gpu_used = instance_spec.instance_type == InstanceType.INSTANCE_TYPE_GPU

            if is_gpu_used or self.build_properties.gpu_compatible:
                result = build_orchestrator.fetch_base_docker_image_name(BaseDockerImageType.GPU)
                self.build_env.docker.base_image = result.base_docker_image_name
            else:
                result = build_orchestrator.fetch_base_docker_image_name(BaseDockerImageType.CPU)
                self.build_env.docker.base_image = result.base_docker_image_name
