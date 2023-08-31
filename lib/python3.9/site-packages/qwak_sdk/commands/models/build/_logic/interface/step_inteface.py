from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from qwak_sdk.commands.models.build._logic.config.config_v1 import ConfigV1
from qwak_sdk.commands.models.build._logic.constant.step_description import BuildPhase
from qwak_sdk.commands.models.build._logic.context import Context

if TYPE_CHECKING:
    from qwak_sdk.commands.models.build import Notifier


class Step(ABC):
    context: Context
    config: ConfigV1
    notifier: Notifier
    build_phase: BuildPhase

    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def execute(self) -> None:
        pass

    def set_notifier(self, notifier: Notifier) -> None:
        self.notifier = notifier
