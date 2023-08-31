from typing import List, Optional

import grpc
from dependency_injector.wiring import Provide
from _qwak_proto.qwak.builds.build_pb2 import BuildInitiator, RemoteBuildSpec
from _qwak_proto.qwak.builds.internal_builds_orchestrator_service_pb2 import (
    BuildInitDetails,
    InternalBuildModelRequest,
)
from _qwak_proto.qwak.builds.internal_builds_orchestrator_service_pb2_grpc import (
    InternalBuildsOrchestratorServiceStub,
)
from qwak.clients.build_orchestrator.build_model_request_getter import (
    _get_build_model_spec,
)
from qwak.inner.di_configuration import QwakContainer
from qwak.exceptions import QwakException


class InternalBuildOrchestratorClient:
    def __init__(
        self,
        grpc_channel_factory=Provide[
            QwakContainer.internal_grpc_channel_for_builds_orchestrator_factory
        ],
        url_overwrite: Optional[str] = None,
        ssl_overwrite: Optional[bool] = None,
    ):
        self.grpc_channel_factory = grpc_channel_factory
        self._internal_builds_orchestrator_stub: Optional[
            InternalBuildsOrchestratorServiceStub
        ] = None
        self.url_overwrite = url_overwrite
        self.ssl_overwrite = ssl_overwrite

    def build_model(
        self,
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
        build_initiator: BuildInitiator = None,
        sdk_version: str = "",
    ):
        """Initiate remote build

        Args:
            verbose: log verbosity level
            git_commit_id: commit id
            resolved_model_url: the url of model
            build_conf: the build configuration
            build_code_path: The code  path saved by qwak
            build_v1_flag:
            build_config_url:
            qwak_sdk_wheel_url:
            qwak_sdk_version_url:
            build_steps: List of the steps the build is comprised from
            build_initiator: Override the initiator of build
            sdk_version: The sdk version to build the

        Raises:
            QwakException: In case of failing to connect the service
        """
        if not self._internal_builds_orchestrator_stub:
            grpc_channel = self.grpc_channel_factory(
                self.url_overwrite, self.ssl_overwrite
            )
            if grpc_channel:
                self._internal_builds_orchestrator_stub = (
                    InternalBuildsOrchestratorServiceStub(grpc_channel)
                )

        if not self._internal_builds_orchestrator_stub:
            raise QwakException("Internal GRPC channel is not available.")

        build_steps = build_steps if build_steps else []

        try:
            build_spec: RemoteBuildSpec = _get_build_model_spec(
                build_conf,
                verbose,
                git_commit_id,
                resolved_model_url,
                build_code_path,
                build_v1_flag,
                build_config_url,
                qwak_sdk_wheel_url,
                qwak_sdk_version_url,
                build_steps,
                sdk_version,
            )
            self._internal_builds_orchestrator_stub.BuildModel(
                InternalBuildModelRequest(
                    build_init_details=BuildInitDetails(
                        build_spec=build_spec, initiator=build_initiator
                    )
                )
            )
        except grpc.RpcError as e:
            message = (
                f"Failed to build model, status [{e.code()}] details [{e.details()}]"
            )
            raise QwakException(message)

        except Exception as e:
            message = f"Failed to build model, details [{e}]"
            raise QwakException(message)
