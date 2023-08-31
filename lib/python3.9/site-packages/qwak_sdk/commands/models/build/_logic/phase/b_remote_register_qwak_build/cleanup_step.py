import os

from qwak_sdk.commands.models.build._logic.interface.step_inteface import Step
from qwak_sdk.commands.models.build._logic.util.step_decorator import (
    cleaning_up_after_build,
)


class CleanupStep(Step):
    def description(self) -> str:
        return "Clean build artifacts"

    def execute(self) -> None:
        self.notifier.debug(
            "Cleanup environment - Deleting file and intermediate images"
        )
        if os.getenv("POD_NAME"):
            self.notifier.debug("Skipping cleanup - Remote executor is temporary")
        else:
            cleaning_up_after_build(self)
