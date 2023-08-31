import json
import re
from typing import List, Optional

import yaml
from _qwak_proto.qwak.builds.build_pb2 import (
    BuildEnv,
    BuildProperties,
    BuildPropertiesV1,
    DockerEnv,
    MemoryUnit,
    ModelUriSpec,
    PythonEnv,
    RemoteBuildSpec,
)
from _qwak_proto.qwak.user_application.common.v0.resources_pb2 import (
    CpuResources,
    GpuResources,
    PodComputeResourceTemplateSpec,
)
from yaml import Loader


def _get_build_model_spec(
    build_conf,
    verbose: int = 3,
    git_commit_id: str = "",
    resolved_model_url: str = "",
    build_code_path: str = "",
    build_v1_flag: bool = False,
    build_config_url: str = "",
    qwak_sdk_wheel_url: str = "",
    qwak_sdk_version_url: str = "",
    build_steps: Optional[List[str]] = None,
    sdk_version: str = "",
) -> RemoteBuildSpec:
    build_spec = RemoteBuildSpec(
        build_properties=BuildProperties(
            build_id=build_conf.build_properties.build_id,
            model_id=build_conf.build_properties.model_id,
            branch=build_conf.build_properties.branch,
            tags=build_conf.build_properties.tags,
            model_uri=ModelUriSpec(
                uri=resolved_model_url or build_conf.build_properties.model_uri.uri,
                git_credentials=build_conf.build_properties.model_uri.git_credentials,
                git_credentials_secret=build_conf.build_properties.model_uri.git_credentials_secret,
                git_branch=build_conf.build_properties.model_uri.git_branch,
                commit_id=git_commit_id,
                main_dir=build_conf.build_properties.model_uri.main_dir,
            ),
        ),
        build_env=BuildEnv(
            docker_env=DockerEnv(
                base_image=build_conf.build_env.docker.base_image,
                assumed_iam_role_arn=build_conf.build_env.docker.assumed_iam_role_arn,
                no_cache=not build_conf.build_env.docker.cache,
                env_vars=build_conf.build_env.docker.env_vars,
            ),
            python_env=PythonEnv(
                git_credentials=build_conf.build_properties.model_uri.git_credentials,
                git_credentials_secret=build_conf.build_properties.model_uri.git_credentials_secret,
                qwak_sdk_version=sdk_version,
            ),
        ),
        verbose=verbose,
        build_code_path=build_code_path,
        build_config=json.dumps(
            yaml.load(build_conf.to_yaml(), Loader=Loader)  # nosec B506
        ),
        build_properties_v1=BuildPropertiesV1(
            build_config_url=build_config_url,
            qwak_sdk_wheel_url=qwak_sdk_wheel_url,
            qwak_sdk_version_url=qwak_sdk_version_url,
        ),
        build_v1_flag=build_v1_flag,
        build_steps=build_steps,
    )

    if build_conf.build_env.remote.resources.instance:
        build_spec.client_pod_compute_resources.template_spec.CopyFrom(
            PodComputeResourceTemplateSpec(
                template_id=build_conf.build_env.remote.resources.instance
            )
        )
    elif build_conf.build_env.remote.resources.gpu_type:
        build_spec.client_pod_compute_resources.gpu_resources.CopyFrom(
            GpuResources(
                gpu_type=build_conf.build_env.remote.resources.gpu_type,
                gpu_amount=build_conf.build_env.remote.resources.gpu_amount,
            )
        )
    else:
        build_spec.client_pod_compute_resources.cpu_resources.CopyFrom(
            CpuResources(
                cpu=build_conf.build_env.remote.resources.cpus,
                memory_amount=int(
                    re.sub(
                        r"\D",
                        "",
                        build_conf.build_env.remote.resources.memory,
                    )
                ),
                memory_units=map_memory_units(
                    build_conf.build_env.remote.resources.memory
                ),
            )
        )

    return build_spec


def map_memory_units(memory):
    memory_unit = re.sub(r"\d+", "", memory)
    if memory_unit == "Gi":
        return MemoryUnit.GIB
    elif memory_unit == "Mib":
        return MemoryUnit.MIB
    else:
        return MemoryUnit.UNKNOWN_MEMORY_UNIT
