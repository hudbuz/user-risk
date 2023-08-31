from dataclasses import dataclass, field
from enum import Enum

from _qwak_proto.qwak.build.v1.build_pb2 import MetricFilter as MetricFilterProto


class MetricOperatorType(Enum):
    METRIC_OPERATOR_TYPE_INVALID = 0
    METRIC_OPERATOR_TYPE_EQUALS = 1
    METRIC_OPERATOR_TYPE_NOT_EQUALS = 2
    METRIC_OPERATOR_TYPE_LESS_THAN = 3
    METRIC_OPERATOR_TYPE_LESS_THAN_EQUALS = 4
    METRIC_OPERATOR_TYPE_GREATER_THAN = 5
    METRIC_OPERATOR_TYPE_GREATER_THAN_EQUALS = 6


@dataclass
class MetricFilter:
    metric_name: str = field(default=None)
    metric_value: float = field(default=None)
    operator: MetricOperatorType = field(
        default=MetricOperatorType.METRIC_OPERATOR_TYPE_INVALID
    )

    def to_proto(self):
        return MetricFilterProto(
            metric_name=self.metric_name,
            metric_value=self.metric_value,
            operator=self.operator.value,
        )

    @staticmethod
    def from_proto(metric_filter_proto: MetricFilterProto):
        return MetricFilter(
            metric_name=metric_filter_proto.metric_name,
            metric_value=metric_filter_proto.metric_value,
            operator=metric_filter_proto.operator,
        )
