from copy import deepcopy

from qwak_sdk import __version__ as qwak_sdk_version
from qwak_sdk.commands.models.build._logic.constant.upload_tag import (
    BUILD_CONFIG_TAG,
    MODEL_CODE_TAG,
    QWAK_SDK_VERSION_TAG,
)
from qwak_sdk.commands.models.build._logic.interface.step_inteface import Step
from qwak_sdk.commands.models.build._logic.util.step_decorator import (
    build_failure_handler,
)


class StartRemoteBuildStep(Step):
    def description(self) -> str:
        return "Start remote build"

    @build_failure_handler()
    def execute(self) -> None:
        self.notifier.info(f"Start remote build - {self.context.build_id}")
        config_copy = deepcopy(self.config)
        config_copy.build_properties.build_id = self.context.build_id
        self.context.client_builds_orchestrator.build_model(
            build_conf=config_copy,
            build_v1_flag=False,
            build_config_url=self.get_download_url(BUILD_CONFIG_TAG),
            qwak_sdk_version_url=self.get_download_url(QWAK_SDK_VERSION_TAG),
            resolved_model_url=self.get_download_url(MODEL_CODE_TAG),
            git_commit_id=self.context.git_commit_id,
            sdk_version=qwak_sdk_version,
        )
        self.notifier.info("Remote build started successfully")

    def get_download_url(self, tag: str) -> str:
        return (
            self.context.client_builds_orchestrator.get_build_versioning_download_url(
                model_id=self.config.build_properties.model_id,
                build_id=self.context.build_id,
                tag=tag,
            ).download_url
        )
