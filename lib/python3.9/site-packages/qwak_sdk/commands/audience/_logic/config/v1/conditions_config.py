from dataclasses import dataclass, field
from typing import List

from _qwak_proto.qwak.audience.v1.audience_pb2 import (
    BinaryCondition,
    BinaryOperatorType,
    Condition,
    UnaryCondition,
    UnaryOperatorType,
)


@dataclass
class UnaryConditionConfig:
    key: str
    operator: str
    operand: str

    def to_condition_api(self) -> Condition:
        return Condition(
            key=self.key,
            unary_condition=UnaryCondition(
                operator=UnaryOperatorType.DESCRIPTOR.values_by_name[
                    self.operator
                ].number,
                operand=self.operand,
            ),
        )


@dataclass
class BinaryConditionConfig:
    key: str
    operator: str
    first_operand: str
    second_operand: str

    def to_condition_api(self) -> Condition:
        return Condition(
            key=self.key,
            binary_condition=BinaryCondition(
                operator=BinaryOperatorType.DESCRIPTOR.values_by_name[
                    self.operator
                ].number,
                first_operand=str(self.first_operand),
                second_operand=str(self.second_operand),
            ),
        )


@dataclass
class ConditionsConfig:
    unary: List[UnaryConditionConfig] = field(default_factory=list)
    binary: List[BinaryConditionConfig] = field(default_factory=list)

    def to_conditions_api(self) -> List[Condition]:
        conditions = [condition.to_condition_api() for condition in self.unary]
        conditions.extend([condition.to_condition_api() for condition in self.binary])
        return conditions
