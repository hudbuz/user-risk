from typing import Any, Callable, Iterable, List

from google.protobuf.json_format import MessageToJson
from tabulate import tabulate


def output_as_json(data: Any):
    print(MessageToJson(data))


def output_as_table(
    data_source: Iterable[Any], data_extractor: Callable, headers: List[str]
):
    data = []
    for item in data_source:
        data.append(data_extractor(item))

    print(tabulate(data, headers=headers))
