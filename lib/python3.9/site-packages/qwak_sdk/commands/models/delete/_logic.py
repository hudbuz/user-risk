from qwak.clients.model_management import ModelsManagementClient


def execute_model_delete(project_id, model_id):
    return ModelsManagementClient().delete_model(project_id, model_id)
