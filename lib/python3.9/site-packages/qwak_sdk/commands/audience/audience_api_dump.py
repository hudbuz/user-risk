from typing import List

from _qwak_proto.qwak.audience.v1.audience_pb2 import (
    Audience,
    AudienceEntry,
    BinaryOperatorType,
    UnaryOperatorType,
)
from google.protobuf.json_format import MessageToJson
from tabulate import tabulate

from qwak_sdk.tools.colors import Color


def audience_to_json(audience: Audience):
    return MessageToJson(audience)


def audience_to_pretty_string(audience_id: str, audience: Audience):
    header = f"{Color.UNDERLINE + Color.BOLD}Audience {audience_id}{Color.END}"
    details_header = f"{Color.UNDERLINE}Details{Color.END}"
    details_table = tabulate(
        tabular_data=[[audience.name, audience.description]],
        headers=["name", "description"],
        tablefmt="pretty",
    )
    conditions_header = f"{Color.UNDERLINE}Conditions{Color.END}"
    conditions_binary_header = f"{Color.CYAN}Binary condition{Color.END}"
    conditions_binary_table = tabulate(
        tabular_data=[
            [
                condition.key,
                BinaryOperatorType.DESCRIPTOR.values_by_number[
                    condition.binary_condition.operator
                ].name,
                condition.binary_condition.first_operand,
                condition.binary_condition.second_operand,
            ]
            for condition in audience.conditions
            if condition.binary_condition.operator
        ],
        headers=["Key", "Operator", "First operand", "Second operand"],
        tablefmt="pretty",
    )
    conditions_unary_header = f"{Color.CYAN}Unary condition{Color.END}"
    conditions_unary_table = tabulate(
        tabular_data=[
            [
                condition.key,
                UnaryOperatorType.DESCRIPTOR.values_by_number[
                    condition.unary_condition.operator
                ].name,
                condition.unary_condition.operand,
            ]
            for condition in audience.conditions
            if condition.unary_condition.operator
        ],
        headers=["Key", "Operator", "Operand"],
        tablefmt="pretty",
    )

    return f"""
{header}
{details_header}
{details_table}

{conditions_header}
{conditions_unary_header}
{conditions_unary_table}

{conditions_binary_header}
{conditions_binary_table}
"""


def audience_entries_to_pretty_string(audience_entries: List[AudienceEntry]) -> str:
    header = f"{Color.BOLD + Color.UNDERLINE}Audiences{Color.END}"
    table = tabulate(
        tabular_data=[[entry.id, entry.audience.name] for entry in audience_entries],
        headers=["Audience ID", "Audience Name"],
        tablefmt="pretty",
    )
    return f"""
{header}
{table}
"""
