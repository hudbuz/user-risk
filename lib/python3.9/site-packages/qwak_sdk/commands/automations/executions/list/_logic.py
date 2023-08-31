from typing import List

from qwak.automations.automation_executions import AutomationExecution
from qwak.clients.automation_management.client import AutomationsManagementClient


def execute_list_executions(automation_id: str) -> List[AutomationExecution]:
    return AutomationsManagementClient().list_executions(automation_id)
