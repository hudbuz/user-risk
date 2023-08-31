from qwak.clients.automation_management.client import AutomationsManagementClient


def delete_automation(automation_id: str):
    client = AutomationsManagementClient()
    return client.delete_automation(automation_id=automation_id)
