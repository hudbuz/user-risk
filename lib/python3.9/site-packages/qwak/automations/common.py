import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from _qwak_proto.qwak.automation.v1.action_pb2 import Action as ActionProto, MemoryUnit
from _qwak_proto.qwak.automation.v1.common_pb2 import MetricThresholdDirection
from _qwak_proto.qwak.user_application.common.v0.resources_pb2 import (
    MemoryUnit as CommonMemoryUnit,
)


@dataclass
class Action(ABC):
    @abstractmethod
    def to_proto(self):
        # abstract method
        pass

    @staticmethod
    @abstractmethod
    def from_proto(message: ActionProto):
        # abstract method
        pass


def map_memory_units(memory: str):
    memory_unit = re.sub(r"\d+", "", memory)
    if memory_unit == "Gi":
        return MemoryUnit.GIB
    elif memory_unit == "Mib":
        return MemoryUnit.MIB
    else:
        return MemoryUnit.UNKNOWN


def map_common_memory_units(memory: str):
    memory_unit = re.sub(r"\d+", "", memory)
    if memory_unit == "Gi":
        return CommonMemoryUnit.GIB
    elif memory_unit == "Mib":
        return CommonMemoryUnit.MIB
    else:
        return CommonMemoryUnit.INVALID_MEMORY_UNIT


def map_memory_units_proto(memory_unit: MemoryUnit):
    if memory_unit == MemoryUnit.MIB:
        return "Mib"
    elif memory_unit == MemoryUnit.GIB:
        return "Gi"
    else:
        return ""


def map_common_memory_units_proto(memory_unit: CommonMemoryUnit):
    if memory_unit == CommonMemoryUnit.MIB:
        return "Mib"
    elif memory_unit == CommonMemoryUnit.GIB:
        return "Gi"
    else:
        return ""


def get_memory_amount(memory: str):
    return int(
        re.sub(
            r"\D",
            "",
            memory,
        )
    )


class ThresholdDirection(Enum):
    ABOVE = 1
    BELOW = 2


threshold_to_proto_mapping = {
    ThresholdDirection.ABOVE: MetricThresholdDirection.ABOVE,
    ThresholdDirection.BELOW: MetricThresholdDirection.BELOW,
}

proto_threshold_to_threshold = {v: k for k, v in threshold_to_proto_mapping.items()}


def map_threshold_direction_to_proto(
    direction: ThresholdDirection,
) -> MetricThresholdDirection:
    return threshold_to_proto_mapping.get(
        direction, MetricThresholdDirection.INVALID_METRIC_DIRECTION
    )


def map_proto_threshold_to_direction(
    direction: MetricThresholdDirection,
) -> ThresholdDirection:
    return proto_threshold_to_threshold.get(direction)
