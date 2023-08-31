from typing import List

from qwak.automations import Automation
from qwak.clients.automation_management.client import AutomationsManagementClient

from qwak_sdk.inner.file_registry import extract_class_objects
from qwak_sdk.inner.tools.cli_tools import ask_yesno
from qwak_sdk.tools.utils import qwak_spinner

DELIMITER = "----------------------------------------"


def register_automations(qwak_python_files: List[str], force: bool):
    """
    Register Automation Entities Objects

    Args:
        qwak_python_files: a list of python files containing qwak package imports
        force: to force
    """
    with qwak_spinner(
        begin_text="Finding Automations to register", print_callback=print
    ):
        qwak_automations: List[Automation] = extract_class_objects(
            qwak_python_files, Automation
        )
    client = AutomationsManagementClient()
    print(f"Found {len(qwak_automations)} Automations")
    for automation, source_file_path in qwak_automations:
        existing_automation = client.get_automation_by_name(automation.name)
        if existing_automation:
            if ask_yesno(
                f"Update existing Automation '{automation.name}' from source file '{source_file_path}'?",
                force,
            ):
                client.update_automation(existing_automation.id, automation.to_proto())
        else:
            if ask_yesno(
                f"Create new Automation '{automation.name}' from source file '{source_file_path}'?",
                force,
            ):
                client.create_automation(automation.to_proto())
    print(DELIMITER)
