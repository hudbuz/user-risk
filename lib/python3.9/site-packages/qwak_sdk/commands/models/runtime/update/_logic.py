from qwak.clients.deployment.client import DeploymentManagementClient
from qwak.clients.model_management import ModelsManagementClient


def execute_runtime_update(model_id, log_level):
    model_uuid = ModelsManagementClient().get_model_uuid(model_id)
    DeploymentManagementClient().update_runtime_configurations(
        model_id=model_id, model_uuid=model_uuid, log_level=log_level
    )
