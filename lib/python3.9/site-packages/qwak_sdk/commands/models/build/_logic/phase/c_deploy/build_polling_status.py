from __future__ import annotations

import time

from _qwak_proto.qwak.build.v1.build_api_pb2 import GetBuildResponse
from _qwak_proto.qwak.build.v1.build_pb2 import BuildStatus

from qwak_sdk.commands.models.build._logic.client_logs.messages import (
    FAILED_REMOTE_BUILD_SUGGESTION,
)
from qwak_sdk.commands.models.build._logic.interface.step_inteface import Step
from qwak_sdk.exceptions.qwak_remote_build_failed import QwakRemoteBuildFailedException


class BuildPollingStatusStep(Step):
    FINITE_BUILD_STATUSES = {
        BuildStatus.REMOTE_BUILD_CANCELLED,
        BuildStatus.REMOTE_BUILD_TIMED_OUT,
        BuildStatus.FAILED,
        BuildStatus.SUCCESSFUL,
    }
    SLEEP_BETWEEN_STATUS_CHECK = 10
    REMOTE_BUILD_FAILURE_EXCEPTION_FORMAT = "Your build failed with status {status}"

    def description(self) -> str:
        return "Polling on Build Status"

    def execute(self) -> None:
        self.notifier.info("Waiting for build to finish")
        status = self.wait_for_finite_build_status()
        if status != BuildStatus.SUCCESSFUL:
            raise QwakRemoteBuildFailedException(
                message=self.REMOTE_BUILD_FAILURE_EXCEPTION_FORMAT.format(
                    status=BuildStatus.Name(status)
                ),
                suggestion=FAILED_REMOTE_BUILD_SUGGESTION.format(
                    build_id=self.context.build_id,
                    model_id=self.context.model_id,
                    project_uuid=self.context.project_uuid,
                ),
            )
        self.notifier.info("Your build finished successfully")

    def wait_for_finite_build_status(self) -> BuildStatus:
        self.notifier.spinner_text(line="Waiting for build to finish")
        while True:
            result: GetBuildResponse = (
                self.context.client_builds_orchestrator.get_build(self.context.build_id)
            )
            status: BuildStatus = result.build.build_status
            if status in self.FINITE_BUILD_STATUSES:
                return status
            self.notifier.debug(f"Build status is currently {BuildStatus.Name(status)}")
            time.sleep(self.SLEEP_BETWEEN_STATUS_CHECK)
