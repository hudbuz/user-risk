from dataclasses import dataclass, field
from enum import Enum

from _qwak_proto.qwak.build.v1.build_pb2 import ParameterFilter as ParameterFilterProto


class ParameterOperatorType(Enum):
    PARAMETER_OPERATOR_TYPE_INVALID = 0
    PARAMETER_OPERATOR_TYPE_EQUALS = 1
    PARAMETER_OPERATOR_TYPE_NOT_EQUALS = 2


@dataclass
class ParameterFilter:
    parameter_name: str = field(default=None)
    parameter_value: str = field(default=None)
    operator: ParameterOperatorType = field(
        default=ParameterOperatorType.PARAMETER_OPERATOR_TYPE_INVALID
    )

    def to_proto(self):
        return ParameterFilterProto(
            parameter_name=self.parameter_name,
            parameter_value=self.parameter_value,
            operator=self.operator.value,
        )

    @staticmethod
    def from_proto(parameter_filter_proto: ParameterFilterProto):
        return ParameterFilter(
            parameter_name=parameter_filter_proto.parameter_name,
            parameter_value=parameter_filter_proto.parameter_value,
            operator=parameter_filter_proto.operator,
        )
