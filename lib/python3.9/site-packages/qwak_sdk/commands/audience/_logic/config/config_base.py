from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from _qwak_proto.qwak.audience.v1.audience_pb2 import Audience


@dataclass
class ConfigBase(ABC):
    spec: Any
    api_version: str = field(default="v1")

    @abstractmethod
    def to_audiences_api(self) -> Audience:
        pass
