from dataclasses import dataclass, field
from typing import List

from qwak_sdk.commands.audience._logic.config.v1.audience_config import AudienceConfig


@dataclass
class Spec:
    audiences: List[AudienceConfig] = field(default_factory=list)
