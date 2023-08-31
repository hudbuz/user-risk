from typing import Optional

from qwak.clients.instance_template.client import InstanceTemplateManagementClient
from qwak.clients.workspace_manager import WorkspaceManagerClient
from qwak.exceptions import QwakException

from qwak_sdk.commands._logic.instance_template import verify_template_id

WORKSPACE_NAME_EMPTY_VALIDATION_MSG = "Workspace name cannot be empty"
WORKSPACE_NAME_LENGTH_VALIDATION_MSG = (
    "Workspace name must be between 3 and 254 characters."
)
MIN_WORKSPACE_NAME_LENGTH = 3
MAX_WORKSPACE_NAME_LENGTH = 254


def _validate_workspace_name(name: str):
    if (name is None) or (len(name) == 0):
        raise QwakException(WORKSPACE_NAME_EMPTY_VALIDATION_MSG)
    if (len(name) < MIN_WORKSPACE_NAME_LENGTH) or (
        len(name) > MAX_WORKSPACE_NAME_LENGTH
    ):
        raise QwakException(WORKSPACE_NAME_LENGTH_VALIDATION_MSG)


def _verify_image_id(image_id: str, workspace_manager_client: WorkspaceManagerClient):
    workspace_images = workspace_manager_client.get_workspace_images().workspace_images
    image_ids = [image.id for image in workspace_images]
    if image_id not in image_ids:
        raise QwakException(
            f"Image {image_id} is not supported. Supported images are: {image_ids}"
        )


def _verify_instance(instance: str):
    instance_template_client = InstanceTemplateManagementClient()
    verify_template_id(instance, instance_template_client)


def validate_id_or_name(workspace_id: Optional[str], workspace_name: Optional[str]):
    if (not workspace_id) and (not workspace_name):
        raise QwakException("Workspace id or name must be provided")
