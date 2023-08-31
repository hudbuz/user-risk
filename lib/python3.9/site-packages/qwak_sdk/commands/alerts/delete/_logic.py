from qwak.clients.alerts_registry import AlertingRegistryClient


def execute_delete_channel(name):
    return AlertingRegistryClient().delete_alerting_channel(channel_name=name)
