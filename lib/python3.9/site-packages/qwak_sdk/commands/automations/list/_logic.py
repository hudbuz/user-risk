from qwak.clients.automation_management.client import AutomationsManagementClient

DELIMITER = "----------------------------------------"


def execute_list_automations():
    client = AutomationsManagementClient()
    return client.list_automations()
