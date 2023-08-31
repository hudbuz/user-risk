from typing import List, Tuple

from qwak.clients.alerts_registry import AlertingRegistryClient
from qwak.clients.alerts_registry.channel import Channel
from tabulate import tabulate


def execute_list_channels():
    alerts_client = AlertingRegistryClient()
    channels: List[Tuple[str, Channel]] = alerts_client.list_alerting_channel()
    columns = ["id", "Name", "Type", "repr"]
    data = []
    for c_id, c in channels:
        data.append([c_id, c.name, type(c.channel_conf).__name__, c.__dict__])
    print(tabulate(data, headers=columns))
